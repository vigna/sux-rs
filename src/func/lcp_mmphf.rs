/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

#![allow(clippy::type_complexity)]

//! LCP-based monotone minimal perfect hash functions.
//!
//! Given *n* keys in sorted order, a monotone minimal perfect hash function
//! maps each key to its rank (0 to *n* − 1). Querying a key not in the
//! original set returns an arbitrary value (same contract as
//! [`VFunc::get`]).
//!
//! [`LcpMmphfInt`] works with any primitive integer type, whereas [`LcpMmphf`]
//! works with any byte-sequence key type (`K: AsRef<[u8]>`). Type aliases
//! [`LcpMmphfStr`] and [`LcpMmphfSliceU8`] are provided for convenience. For
//! the byte-sequence variant, keys must not contain zeros, as a virtual zero
//! byte is appended internally to ensure prefix-freeness. Alternatively,
//! they must be prefix-free, in which case they can contain zeros.
//!
//! These structures implement the [`TryIntoUnaligned`] trait, allowing them to
//! be converted into (usually faster) structures using unaligned access.
//!
//! # Implementation details
//!
//! The keys are divided into buckets. For each bucket, the longest common
//! bit-level prefix (LCP) shared by all consecutive pairs within the bucket is
//! computed. The rank of a key is then reconstructed as `bucket * bucket_size +
//! offset`, where the bucket is determined by the LCP (computed from its
//! length) and the offset is stored directly. The mapping from a key to the
//! length of the prefix of its bucket and its offset within the bucket is
//! stored in a [`VFunc`]. Another [`VFunc`] maps each LCP bit-prefix to its
//! bucket index.
//!
//! # References
//!
//! Djamal Belazzougui, Paolo Boldi, Rasmus Pagh, and Sebastiano Vigna.
//! [Monotone minimal perfect hashing: Searching a sorted table with O(1)
//! accesses]. In *Proceedings of the 20th Annual ACM-SIAM Symposium On
//! Discrete Mathematics (SODA)*, pages 785−794, New York, 2009. ACM Press.
//!
//! [Monotone minimal perfect hashing: Searching a sorted table with O(1) accesses]: https://dl.acm.org/doi/10.5555/1496770.1496856

use crate::bits::BitFieldVec;
use crate::func::VFunc;
use crate::func::shard_edge::{Fuse3NoShards, FuseLge3Shards, ShardEdge};
use crate::traits::TryIntoUnaligned;
use crate::traits::Unaligned;
use crate::utils::*;
use mem_dbg::*;
use num_primitive::PrimitiveInteger;
use std::borrow::Borrow;
use value_traits::slices::SliceByValue;
use xxhash_rust::xxh3;

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
#[derive(Debug, Clone, Copy, MemSize, MemDbg)]
#[mem_size(flat)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct IntBitPrefix<K> {
    /// The original integer value.
    value: K,
    /// Number of significant leading bits.
    bit_length: usize,
}

impl<K: PrimitiveInteger> IntBitPrefix<K> {
    /// Creates a new integer bit prefix.
    #[inline]
    pub fn new(value: K, bit_length: usize) -> Self {
        debug_assert!(bit_length <= K::BITS as usize);
        Self { value, bit_length }
    }

    /// Returns the value with the bottom `K::BITS - bit_length` bits
    /// masked out. When `bit_length == 0` (shift by `K::BITS` would
    /// overflow), returns `K::MIN`; this is harmless because the hash
    /// includes `bit_length` for disambiguation.
    ///
    /// Uses `K::MIN | K::MAX` as the all-ones value so that the mask
    /// is correct for both signed types (where `K::MAX` lacks the MSB)
    /// and unsigned types (where `K::MIN | K::MAX == K::MAX`).
    ///
    /// The `match` on `checked_shl` should compile to a branchless
    /// conditional move.
    #[inline]
    fn masked_value(&self) -> K {
        let all_ones = K::MIN | K::MAX;
        match all_ones.checked_shl(K::BITS - self.bit_length as u32) {
            Some(m) => self.value & m,
            None => K::MIN, // bit_length == 0
        }
    }
}

/// Packs significant bits and bit_length into a contiguous stack buffer
/// for one-shot xxh3 hashing. Returns the number of bytes written.
/// Buffer must be at least `size_of::<K>() + size_of::<usize>()` bytes.
#[inline]
pub(crate) fn pack_int_bit_prefix<K: PrimitiveInteger>(
    bp: &IntBitPrefix<K>,
    buf: &mut [u8],
) -> usize {
    let val: K::Bytes = bp.masked_value().to_ne_bytes();
    let val = val.borrow() as &[u8];
    let len = bp.bit_length.to_ne_bytes();
    let n = val.len() + len.len();
    buf[..val.len()].copy_from_slice(val);
    buf[val.len()..n].copy_from_slice(&len);
    n
}

impl<K: PrimitiveInteger> ToSig<[u64; 1]> for IntBitPrefix<K> {
    #[inline]
    fn to_sig(key: impl Borrow<Self>, seed: u64) -> [u64; 1] {
        let mut buf = [0u8; 24];
        let n = pack_int_bit_prefix(key.borrow(), &mut buf);
        let mut hasher = xxh3::Xxh3::with_seed(seed);
        hasher.update(&buf[..n]);
        <[u64; 1]>::from_hasher(&hasher)
    }
}

#[cfg(feature = "rayon")]
mod build {
    use std::ops::{BitXor, BitXorAssign};

    use super::*;
    use crate::func::VBuilder;
    use anyhow::{Result, bail};
    use dsi_progress_logger::ProgressLog;
    use lender::*;
    use rdst::RadixKey;

    /// Returns the number of leading bits shared by two integers of the same
    /// type, compared MSB-first (big-endian bit order).
    #[inline(always)]
    pub(crate) fn lcp_bits<K: PrimitiveInteger>(a: K, b: K) -> usize {
        (a ^ b).leading_zeros() as usize
    }

    /// Computes the log2 of bucket size from *n*.
    pub(crate) fn log2_bucket_size(n: usize) -> usize {
        if n <= 1 {
            return 0;
        }
        let ln_n = (n as f64).ln();
        let theoretical = 1.0 + 1.11 * f64::ln(2.0) + ln_n - (1.0 + ln_n).ln();
        let theoretical = theoretical.ceil().max(1.0) as usize;
        theoretical.next_power_of_two().ilog2() as usize
    }

    impl<
        K: PrimitiveInteger + ToSig<S0> + std::fmt::Debug + Send + Sync + Copy + Ord,
        S0: Sig + Send + Sync,
        E0: ShardEdge<S0, 3> + MemSize + mem_dbg::FlatType,
        S1: Sig + Send + Sync,
        E1: ShardEdge<S1, 3> + MemSize + mem_dbg::FlatType,
    > LcpMmphfInt<K, BitFieldVec<Box<[usize]>>, S0, E0, S1, E1>
    where
        IntBitPrefix<K>: ToSig<S1>,
        SigVal<S0, usize>: RadixKey,
        SigVal<E0::LocalSig, usize>: BitXor + BitXorAssign,
        SigVal<S1, usize>: RadixKey,
        SigVal<E1::LocalSig, usize>: BitXor + BitXorAssign,
    {
        /// Creates a new LCP-based monotone minimal perfect hash function
        /// for integers using default [`VBuilder`] settings.
        ///
        /// This is a convenience wrapper around [`try_new_with_builder`].
        /// Use that method if you need to configure construction parameters
        /// such as offline mode, thread count, or sharding overhead.
        ///
        /// The keys must be provided in strictly increasing order.
        ///
        /// Keys must be provided as a [`FallibleRewindableLender`]. The [`lenders`]
        /// module provides easy ways to build such lenders.
        ///
        /// If keys are available as a slice, [`try_par_new`] parallelizes
        /// the hash computation for faster construction.
        ///
        /// # Examples
        ///
        /// ```rust
        /// # #[cfg(feature = "rayon")]
        /// # fn main() -> anyhow::Result<()> {
        /// # use sux::func::LcpMmphfInt;
        /// # use sux::traits::TryIntoUnaligned;
        /// # use dsi_progress_logger::no_logging;
        /// # use sux::utils::FromSlice;
        /// let keys: Vec<u64> = vec![10, 20, 30, 40, 50];
        /// let func =
        ///     LcpMmphfInt::<u64>::try_new(FromSlice::new(&keys), keys.len(), no_logging![])?.try_into_unaligned()?;
        ///
        /// for (i, &key) in keys.iter().enumerate() {
        ///     assert_eq!(func.get(key), i);
        /// }
        /// # Ok(())
        /// # }
        /// # #[cfg(not(feature = "rayon"))]
        /// # fn main() {}
        /// ```
        ///
        /// [`try_new_with_builder`]: Self::try_new_with_builder
        /// [`try_par_new`]: Self::try_par_new
        pub fn try_new(
            keys: impl FallibleRewindableLender<
                RewindError: std::error::Error + Send + Sync + 'static,
                Error: std::error::Error + Send + Sync + 'static,
            > + for<'lend> FallibleLending<'lend, Lend = &'lend K>,
            n: usize,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self> {
            Self::try_new_with_builder(keys, n, VBuilder::default(), pl)
        }

        /// Creates a new LCP-based monotone minimal perfect hash function
        /// for integers using the given [`VBuilder`] configuration.
        ///
        /// The builder controls construction parameters such as [offline
        /// mode], [thread count], [sharding overhead], and [PRNG seed].
        ///
        /// The keys must be provided in strictly increasing order.
        ///
        /// Keys must be provided as a [`FallibleRewindableLender`]. The [`lenders`]
        /// module provides easy ways to build such lenders.
        ///
        /// See also [`try_par_new_with_builder`] for parallel hash
        /// computation from slices.
        ///
        /// # Examples
        ///
        /// ```rust
        /// # #[cfg(feature = "rayon")]
        /// # fn main() -> anyhow::Result<()> {
        /// # use sux::func::{LcpMmphfInt, VBuilder};
        /// # use sux::traits::TryIntoUnaligned;
        /// # use dsi_progress_logger::no_logging;
        /// # use sux::utils::FromSlice;
        /// let keys: Vec<u64> = vec![10, 20, 30, 40, 50];
        /// let func = LcpMmphfInt::<u64>::try_new_with_builder(
        ///     FromSlice::new(&keys),
        ///     keys.len(),
        ///     VBuilder::default().offline(true),
        ///     no_logging![],
        /// )?.try_into_unaligned()?;
        ///
        /// for (i, &key) in keys.iter().enumerate() {
        ///     assert_eq!(func.get(key), i);
        /// }
        /// # Ok(())
        /// # }
        /// # #[cfg(not(feature = "rayon"))]
        /// # fn main() {}
        /// ```
        ///
        /// [offline mode]: VBuilder::offline
        /// [thread count]: VBuilder::max_num_threads
        /// [sharding overhead]: VBuilder::eps
        /// [PRNG seed]: VBuilder::seed
        /// [`try_par_new_with_builder`]: Self::try_par_new_with_builder
        pub fn try_new_with_builder(
            keys: impl FallibleRewindableLender<
                RewindError: std::error::Error + Send + Sync + 'static,
                Error: std::error::Error + Send + Sync + 'static,
            > + for<'lend> FallibleLending<'lend, Lend = &'lend K>,
            n: usize,
            builder: VBuilder<BitFieldVec<Box<[usize]>>, S0, E0>,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self> {
            Self::try_new_inner(keys, n, builder, pl).map(|(mmphf, _)| mmphf)
        }

        /// Internal constructor accepting a [`VBuilder`] and returning
        /// the keys lender (for rewinding).
        pub(crate) fn try_new_inner<
            L: FallibleRewindableLender<
                    RewindError: std::error::Error + Send + Sync + 'static,
                    Error: std::error::Error + Send + Sync + 'static,
                > + for<'lend> FallibleLending<'lend, Lend = &'lend K>,
        >(
            mut keys: L,
            n: usize,
            builder: VBuilder<BitFieldVec<Box<[usize]>>, S0, E0>,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<(Self, L)> {
            if n == 0 {
                return Ok((
                    Self {
                        n: 0,
                        log2_bucket_size: 0,
                        offset_lcp_length: VFunc::empty(),
                        lcp2bucket: VFunc::empty(),
                    },
                    keys,
                ));
            }

            let log2_bs = log2_bucket_size(n);
            let bucket_size = 1usize << log2_bs;
            let bucket_mask = bucket_size - 1;
            let num_buckets = n.div_ceil(bucket_size);
            let lcp2bucket_builder = VBuilder::default()
                .set_from(&builder)
                .expected_num_keys(num_buckets);

            pl.info(format_args!(
                "Bucket size: 2^{log2_bs} = {bucket_size} ({num_buckets} buckets for {n} keys)"
            ));

            // -- First pass: compute bit-level LCPs --

            let mut lcp_bit_lengths: Vec<usize> = Vec::with_capacity(num_buckets);
            let mut bucket_first_keys: Vec<K> = Vec::with_capacity(num_buckets);

            let mut prev_key: Option<K> = None;
            let mut curr_lcp_bits: usize = 0;
            let mut i = 0usize;

            while let Some(key) = keys.next()? {
                let key: K = *key;

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
                    curr_lcp_bits = K::BITS as usize;
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

            let (offset_lcp_length, keys) =
                builder.expected_num_keys(n).try_build_func::<K, K, _, _>(
                    keys,
                    FromCloneableIntoIterator::new((0..n).map(|idx| {
                        (lcp_bit_lengths[idx >> log2_bs] << log2_bs) | (idx & bucket_mask)
                    })),
                    BitFieldVec::<Box<[usize]>>::new_padded,
                    pl,
                )?;

            // Sequential: num_buckets is small and we avoid materializing the key set
            pl.info(format_args!(
                "Building LCP prefix → bucket map ({num_buckets} buckets)..."
            ));
            let lcp2bucket =
                <VFunc<IntBitPrefix<K>, BitFieldVec<Box<[usize]>>, S1, E1>>::try_new_with_builder(
                    FromCloneableIntoIterator::new((0..num_buckets).map(|b| {
                        IntBitPrefix::new(bucket_first_keys[b] ^ K::MIN, lcp_bit_lengths[b])
                    })),
                    FromCloneableIntoIterator::new(0..num_buckets),
                    lcp2bucket_builder,
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

            Ok((result, keys))
        }

        /// Creates a new LCP-based monotone minimal perfect hash function
        /// for integers from a slice, using parallel hash computation and
        /// default [`VBuilder`] settings.
        ///
        /// This is the parallel counterpart of [`try_new`]. It is a
        /// convenience wrapper around [`try_par_new_with_builder`] with
        /// `VBuilder::default()`.
        ///
        /// The keys must be provided in strictly increasing order.
        ///
        /// If keys are produced sequentially (e.g., from a file), use
        /// [`try_new`] instead.
        ///
        /// # Examples
        ///
        /// ```rust
        /// # #[cfg(feature = "rayon")]
        /// # fn main() -> anyhow::Result<()> {
        /// # use sux::func::LcpMmphfInt;
        /// # use sux::traits::TryIntoUnaligned;
        /// # use dsi_progress_logger::no_logging;
        /// let keys: Vec<u64> = vec![10, 20, 30, 40, 50];
        /// let func =
        ///     LcpMmphfInt::<u64>::try_par_new(&keys, no_logging![])?.try_into_unaligned()?;
        ///
        /// for (i, &key) in keys.iter().enumerate() {
        ///     assert_eq!(func.get(key), i);
        /// }
        /// # Ok(())
        /// # }
        /// # #[cfg(not(feature = "rayon"))]
        /// # fn main() {}
        /// ```
        ///
        /// [`try_new`]: Self::try_new
        /// [`try_par_new_with_builder`]: Self::try_par_new_with_builder
        pub fn try_par_new(
            keys: &[K],
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self> {
            Self::try_par_new_with_builder(keys, VBuilder::default(), pl)
        }

        /// Creates a new LCP-based monotone minimal perfect hash function
        /// for integers from a slice, using parallel hash computation and
        /// the given [`VBuilder`] configuration.
        ///
        /// This is the parallel counterpart of [`try_new_with_builder`].
        ///
        /// The keys must be provided in strictly increasing order.
        ///
        /// If keys are produced sequentially (e.g., from a file), use
        /// [`try_new_with_builder`] instead.
        ///
        /// # Examples
        ///
        /// ```rust
        /// # #[cfg(feature = "rayon")]
        /// # fn main() -> anyhow::Result<()> {
        /// # use sux::func::{LcpMmphfInt, VBuilder};
        /// # use sux::traits::TryIntoUnaligned;
        /// # use dsi_progress_logger::no_logging;
        /// let keys: Vec<u64> = vec![10, 20, 30, 40, 50];
        /// let func = LcpMmphfInt::<u64>::try_par_new_with_builder(
        ///     &keys,
        ///     VBuilder::default().offline(true),
        ///     no_logging![],
        /// )?.try_into_unaligned()?;
        ///
        /// for (i, &key) in keys.iter().enumerate() {
        ///     assert_eq!(func.get(key), i);
        /// }
        /// # Ok(())
        /// # }
        /// # #[cfg(not(feature = "rayon"))]
        /// # fn main() {}
        /// ```
        ///
        /// [`try_new_with_builder`]: Self::try_new_with_builder
        pub fn try_par_new_with_builder(
            keys: &[K],
            builder: VBuilder<BitFieldVec<Box<[usize]>>, S0, E0>,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self> {
            Self::try_par_new_inner(keys, builder, pl)
        }

        /// Internal parallel constructor for integer keys.
        pub(crate) fn try_par_new_inner(
            keys: &[K],
            builder: VBuilder<BitFieldVec<Box<[usize]>>, S0, E0>,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self> {
            let n = keys.len();
            if n == 0 {
                return Ok(Self {
                    n: 0,
                    log2_bucket_size: 0,
                    offset_lcp_length: VFunc::empty(),
                    lcp2bucket: VFunc::empty(),
                });
            }

            let log2_bs = log2_bucket_size(n);
            let bucket_size = 1usize << log2_bs;
            let bucket_mask = bucket_size - 1;
            let num_buckets = n.div_ceil(bucket_size);
            let lcp2bucket_builder = VBuilder::default()
                .set_from(&builder)
                .expected_num_keys(num_buckets);

            pl.info(format_args!(
                "Bucket size: 2^{log2_bs} = {bucket_size} ({num_buckets} buckets for {n} keys)"
            ));

            // -- Sequential pass: compute bit-level LCPs --

            let mut lcp_bit_lengths: Vec<usize> = Vec::with_capacity(num_buckets);
            let mut bucket_first_keys: Vec<K> = Vec::with_capacity(num_buckets);

            let mut prev_key: Option<K> = None;
            let mut curr_lcp_bits: usize = 0;

            for (i, &key) in keys.iter().enumerate() {
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
                    curr_lcp_bits = K::BITS as usize;
                } else {
                    curr_lcp_bits = curr_lcp_bits.min(lcp_bits(key, prev_key.unwrap()));
                }

                prev_key = Some(key);
            }

            lcp_bit_lengths.push(curr_lcp_bits);
            assert_eq!(lcp_bit_lengths.len(), num_buckets);

            // -- Build offset_lcp_length VFunc (parallel) --

            pl.info(format_args!(
                "Building key → (LCP length, offset) map (parallel)..."
            ));
            let offset_lcp_length = builder.expected_num_keys(n).try_par_populate_and_build(
                keys,
                &|i| (lcp_bit_lengths[i >> log2_bs] << log2_bs) | (i & bucket_mask),
                &mut |builder, seed, mut store, max_value, _num_keys, pl, _state: &mut ()| {
                    builder.bit_width = max_value.bit_len() as usize;
                    let data = BitFieldVec::<Box<[usize]>>::new_padded(
                        builder.bit_width,
                        builder.shard_edge.num_vertices() * builder.shard_edge.num_shards(),
                    );
                    let func = builder.try_build_from_shard_iter(
                        seed,
                        data,
                        store.drain(),
                        &|_, sv| sv.val,
                        &|_| {},
                        pl,
                    )?;
                    Ok(func)
                },
                pl,
                (),
            )?;

            // Sequential: num_buckets is small and we avoid materializing the key set
            pl.info(format_args!(
                "Building LCP prefix → bucket map ({num_buckets} buckets)..."
            ));
            let lcp2bucket =
                <VFunc<IntBitPrefix<K>, BitFieldVec<Box<[usize]>>, S1, E1>>::try_new_with_builder(
                    FromCloneableIntoIterator::new((0..num_buckets).map(|b| {
                        IntBitPrefix::new(bucket_first_keys[b] ^ K::MIN, lcp_bit_lengths[b])
                    })),
                    FromCloneableIntoIterator::new(0..num_buckets),
                    lcp2bucket_builder,
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

            Ok(result)
        }
    }

    use crate::utils::lcp_len;

    /// Returns the number of leading bits that are identical in two byte
    /// slices, after conceptually appending a NUL byte to each. Since keys
    /// must not contain NUL, two distinct prefix-related strings are
    /// guaranteed to diverge at the NUL position.
    ///
    /// The implementation first compares the string bytes (using the
    /// vectorized [`lcp_len`]). If one string is a prefix of the other, the
    /// virtual NUL (0x00) XOR the next byte of the longer string yields that
    /// byte itself, and `leading_zeros` gives the additional shared bits.
    ///
    /// If `DISTINCT` is `true`, the two strings are assumed to be distinct
    /// and the identical-string case is skipped (calling with identical
    /// strings is undefined behavior in debug builds and an out-of-bounds
    /// access in release builds). If `DISTINCT` is `false`, identical
    /// strings return `len * 8 + 8` (all bits match, including the
    /// virtual NUL).
    pub(crate) fn lcp_bits_nul<const DISTINCT: bool>(a: &[u8], b: &[u8]) -> usize {
        let min_len = a.len().min(b.len());
        let pos = lcp_len(&a[..min_len], &b[..min_len]);

        if pos < min_len {
            // Mismatch within the common part — fast path.
            return pos * 8 + (a[pos] ^ b[pos]).leading_zeros() as usize;
        }

        if !DISTINCT && a.len() == b.len() {
            // Identical keys (including virtual NUL). The virtual NUL is
            // 0x00 for both, so all bits match.
            return min_len * 8 + 8;
        }

        // One string is a proper prefix of the other. The virtual NUL
        // after the shorter string diverges from the next byte of the
        // longer string (which is guaranteed non-NUL).
        let longer = if a.len() > b.len() { a } else { b };
        debug_assert!(longer.len() > min_len);
        min_len * 8 + longer[min_len].leading_zeros() as usize
    }

    impl<
        K: ?Sized + AsRef<[u8]> + ToSig<S0> + std::fmt::Debug,
        S0: Sig + Send + Sync,
        E0: ShardEdge<S0, 3> + MemSize + mem_dbg::FlatType,
        S1: Sig + Send + Sync,
        E1: ShardEdge<S1, 3> + MemSize + mem_dbg::FlatType,
    > LcpMmphf<K, BitFieldVec<Box<[usize]>>, S0, E0, S1, E1>
    where
        BitPrefix: ToSig<S1>,
        SigVal<S0, usize>: RadixKey,
        SigVal<E0::LocalSig, usize>: BitXor + BitXorAssign,
        SigVal<S1, usize>: RadixKey,
        SigVal<E1::LocalSig, usize>: BitXor + BitXorAssign,
    {
        /// Creates a new LCP-based monotone minimal perfect hash function for
        /// byte-sequence keys using default [`VBuilder`] settings.
        ///
        /// This is a convenience wrapper around [`try_new_with_builder`].
        /// Use that method if you need to configure construction parameters
        /// such as offline mode, thread count, or sharding overhead.
        ///
        /// The keys must be in strictly increasing lexicographic order.
        /// The lender may yield references to any type `B` that borrows
        /// as `K` (e.g., `&String` for `K = str`, `&Vec<u8>` for
        /// `K = [u8]`).
        ///
        /// Keys must be provided as a [`FallibleRewindableLender`]. The [`lenders`]
        /// module provides easy ways to build such lenders.
        ///
        /// If keys are available as a slice, [`try_par_new`] parallelizes
        /// the hash computation for faster construction.
        ///
        /// # Examples
        ///
        /// ```rust
        /// # #[cfg(feature = "rayon")]
        /// # fn main() -> anyhow::Result<()> {
        /// # use sux::func::LcpMmphfStr;
        /// # use sux::traits::TryIntoUnaligned;
        /// # use dsi_progress_logger::no_logging;
        /// # use sux::utils::FromSlice;
        /// let keys = vec!["a", "b", "c", "d", "e"];
        /// let func =
        ///     <LcpMmphfStr>::try_new(FromSlice::new(&keys), keys.len(), no_logging![])?.try_into_unaligned()?;
        ///
        /// for (i, &key) in keys.iter().enumerate() {
        ///     assert_eq!(func.get(key), i);
        /// }
        /// # Ok(())
        /// # }
        /// # #[cfg(not(feature = "rayon"))]
        /// # fn main() {}
        /// ```
        ///
        /// [`try_new_with_builder`]: Self::try_new_with_builder
        /// [`try_par_new`]: Self::try_par_new
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

        /// Creates a new LCP-based monotone minimal perfect hash function for
        /// byte-sequence keys using the given [`VBuilder`] configuration.
        ///
        /// The builder controls construction parameters such as [offline
        /// mode], [thread count], [sharding overhead], and [PRNG seed].
        ///
        /// The keys must be in strictly increasing lexicographic order.
        ///
        /// Keys must be provided as a [`FallibleRewindableLender`]. The [`lenders`]
        /// module provides easy ways to build such lenders.
        ///
        /// See also [`try_par_new_with_builder`] for parallel hash
        /// computation from slices.
        ///
        /// # Examples
        ///
        /// ```rust
        /// # #[cfg(feature = "rayon")]
        /// # fn main() -> anyhow::Result<()> {
        /// # use sux::func::{LcpMmphfStr, VBuilder};
        /// # use sux::traits::TryIntoUnaligned;
        /// # use dsi_progress_logger::no_logging;
        /// # use sux::utils::FromSlice;
        /// let keys = vec!["a", "b", "c", "d", "e"];
        /// let func = <LcpMmphfStr>::try_new_with_builder(
        ///     FromSlice::new(&keys),
        ///     keys.len(),
        ///     VBuilder::default().offline(true),
        ///     no_logging![],
        /// )?.try_into_unaligned()?;
        ///
        /// for (i, &key) in keys.iter().enumerate() {
        ///     assert_eq!(func.get(key), i);
        /// }
        /// # Ok(())
        /// # }
        /// # #[cfg(not(feature = "rayon"))]
        /// # fn main() {}
        /// ```
        ///
        /// [offline mode]: VBuilder::offline
        /// [thread count]: VBuilder::max_num_threads
        /// [sharding overhead]: VBuilder::eps
        /// [PRNG seed]: VBuilder::seed
        /// [`try_par_new_with_builder`]: Self::try_par_new_with_builder
        pub fn try_new_with_builder<B: ?Sized + AsRef<[u8]> + Borrow<K>>(
            keys: impl FallibleRewindableLender<
                RewindError: std::error::Error + Send + Sync + 'static,
                Error: std::error::Error + Send + Sync + 'static,
            > + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
            n: usize,
            builder: VBuilder<BitFieldVec<Box<[usize]>>, S0, E0>,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self> {
            Self::try_new_inner(keys, n, builder, pl).map(|(mmphf, _)| mmphf)
        }

        /// Internal constructor accepting a [`VBuilder`] and returning
        /// the keys lender (for rewinding).
        pub(crate) fn try_new_inner<
            B: ?Sized + AsRef<[u8]> + Borrow<K>,
            L: FallibleRewindableLender<
                    RewindError: std::error::Error + Send + Sync + 'static,
                    Error: std::error::Error + Send + Sync + 'static,
                > + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
        >(
            mut keys: L,
            n: usize,
            builder: VBuilder<BitFieldVec<Box<[usize]>>, S0, E0>,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<(Self, L)> {
            if n == 0 {
                return Ok((
                    Self {
                        n: 0,
                        log2_bucket_size: 0,
                        offset_lcp_length: VFunc::empty(),
                        lcp2bucket: VFunc::empty(),
                    },
                    keys,
                ));
            }

            let log2_bs = log2_bucket_size(n);
            let bucket_size = 1usize << log2_bs;
            let bucket_mask = bucket_size - 1;
            let num_buckets = n.div_ceil(bucket_size);
            let lcp2bucket_builder = VBuilder::default()
                .set_from(&builder)
                .expected_num_keys(num_buckets);

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
                    curr_lcp_bits = curr_lcp_bits.min(lcp_bits_nul::<true>(key_bytes, &prev_key));
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

            let (offset_lcp_length, keys) =
                builder.expected_num_keys(n).try_build_func::<K, B, _, _>(
                    keys,
                    FromCloneableIntoIterator::new((0..n).map(|idx| {
                        (lcp_bit_lengths[idx >> log2_bs] << log2_bs) | (idx & bucket_mask)
                    })),
                    BitFieldVec::<Box<[usize]>>::new_padded,
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

            // Sequential: num_buckets is small and we avoid materializing the key set
            let lcp2bucket =
                <VFunc<BitPrefix, BitFieldVec<Box<[usize]>>, S1, E1>>::try_new_with_builder(
                    FromCloneableIntoIterator::new(
                        (0..num_buckets)
                            .map(|b| BitPrefix::new(&extended_first_keys[b], lcp_bit_lengths[b])),
                    ),
                    FromCloneableIntoIterator::new(0..num_buckets),
                    lcp2bucket_builder,
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

            Ok((result, keys))
        }

        /// Creates a new LCP-based monotone minimal perfect hash function for
        /// byte-sequence keys from a slice, using parallel hash computation and
        /// default [`VBuilder`] settings.
        ///
        /// This is the parallel counterpart of [`try_new`]. It is a
        /// convenience wrapper around [`try_par_new_with_builder`] with
        /// `VBuilder::default()`.
        ///
        /// The keys must be in strictly increasing lexicographic order.
        ///
        /// If keys are produced sequentially (e.g., from a file), use
        /// [`try_new`] instead.
        ///
        /// # Examples
        ///
        /// ```rust
        /// # #[cfg(feature = "rayon")]
        /// # fn main() -> anyhow::Result<()> {
        /// # use sux::func::LcpMmphfStr;
        /// # use sux::traits::TryIntoUnaligned;
        /// # use dsi_progress_logger::no_logging;
        /// let keys = vec!["a", "b", "c", "d", "e"];
        /// let func =
        ///     <LcpMmphfStr>::try_par_new(&keys, no_logging![])?.try_into_unaligned()?;
        ///
        /// for (i, &key) in keys.iter().enumerate() {
        ///     assert_eq!(func.get(key), i);
        /// }
        /// # Ok(())
        /// # }
        /// # #[cfg(not(feature = "rayon"))]
        /// # fn main() {}
        /// ```
        ///
        /// [`try_new`]: Self::try_new
        /// [`try_par_new_with_builder`]: Self::try_par_new_with_builder
        pub fn try_par_new<B: AsRef<[u8]> + Borrow<K> + Sync>(
            keys: &[B],
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self>
        where
            K: Sync,
        {
            Self::try_par_new_with_builder(keys, VBuilder::default(), pl)
        }

        /// Creates a new LCP-based monotone minimal perfect hash function for
        /// byte-sequence keys from a slice, using parallel hash computation and the
        /// given [`VBuilder`] configuration.
        ///
        /// This is the parallel counterpart of [`try_new_with_builder`].
        ///
        /// The keys must be in strictly increasing lexicographic order.
        ///
        /// If keys are produced sequentially (e.g., from a file), use
        /// [`try_new_with_builder`] instead.
        ///
        /// # Examples
        ///
        /// ```rust
        /// # #[cfg(feature = "rayon")]
        /// # fn main() -> anyhow::Result<()> {
        /// # use sux::func::{LcpMmphfStr, VBuilder};
        /// # use sux::traits::TryIntoUnaligned;
        /// # use dsi_progress_logger::no_logging;
        /// let keys = vec!["a", "b", "c", "d", "e"];
        /// let func = <LcpMmphfStr>::try_par_new_with_builder(
        ///     &keys,
        ///     VBuilder::default().offline(true),
        ///     no_logging![],
        /// )?.try_into_unaligned()?;
        ///
        /// for (i, &key) in keys.iter().enumerate() {
        ///     assert_eq!(func.get(key), i);
        /// }
        /// # Ok(())
        /// # }
        /// # #[cfg(not(feature = "rayon"))]
        /// # fn main() {}
        /// ```
        ///
        /// [`try_new_with_builder`]: Self::try_new_with_builder
        pub fn try_par_new_with_builder<B: AsRef<[u8]> + Borrow<K> + Sync>(
            keys: &[B],
            builder: VBuilder<BitFieldVec<Box<[usize]>>, S0, E0>,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self>
        where
            K: Sync,
        {
            Self::try_par_new_inner(keys, builder, pl)
        }

        /// Internal parallel constructor for byte-sequence keys.
        pub(crate) fn try_par_new_inner<B: AsRef<[u8]> + Borrow<K> + Sync>(
            keys: &[B],
            builder: VBuilder<BitFieldVec<Box<[usize]>>, S0, E0>,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self>
        where
            K: Sync,
        {
            let n = keys.len();
            if n == 0 {
                return Ok(Self {
                    n: 0,
                    log2_bucket_size: 0,
                    offset_lcp_length: VFunc::empty(),
                    lcp2bucket: VFunc::empty(),
                });
            }

            let log2_bs = log2_bucket_size(n);
            let bucket_size = 1usize << log2_bs;
            let bucket_mask = bucket_size - 1;
            let num_buckets = n.div_ceil(bucket_size);
            let lcp2bucket_builder = VBuilder::default()
                .set_from(&builder)
                .expected_num_keys(num_buckets);

            pl.info(format_args!(
                "Bucket size: 2^{log2_bs} = {bucket_size} ({num_buckets} buckets for {n} keys)"
            ));

            // -- Sequential pass: compute bit-level LCPs --

            let mut lcp_bit_lengths: Vec<usize> = Vec::with_capacity(num_buckets);
            let mut bucket_first_keys: Vec<Vec<u8>> = Vec::with_capacity(num_buckets);

            let mut prev_key: Vec<u8> = Vec::new();
            let mut curr_lcp_bits: usize = 0;

            for (i, key) in keys.iter().enumerate() {
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
                    curr_lcp_bits = (key_bytes.len() + 1) * 8;
                } else {
                    curr_lcp_bits = curr_lcp_bits.min(lcp_bits_nul::<true>(key_bytes, &prev_key));
                }

                prev_key.clear();
                prev_key.extend_from_slice(key_bytes);
            }

            lcp_bit_lengths.push(curr_lcp_bits);
            assert_eq!(lcp_bit_lengths.len(), num_buckets);

            // -- Build offset_lcp_length VFunc (parallel) --

            pl.info(format_args!(
                "Building key → (LCP length, offset) map (parallel)..."
            ));
            let offset_lcp_length = builder.expected_num_keys(n).try_par_populate_and_build(
                keys,
                &|i| (lcp_bit_lengths[i >> log2_bs] << log2_bs) | (i & bucket_mask),
                &mut |builder, seed, mut store, max_value, _num_keys, pl, _state: &mut ()| {
                    builder.bit_width = max_value.bit_len() as usize;
                    let data = BitFieldVec::<Box<[usize]>>::new_padded(
                        builder.bit_width,
                        builder.shard_edge.num_vertices() * builder.shard_edge.num_shards(),
                    );
                    let func = builder.try_build_from_shard_iter(
                        seed,
                        data,
                        store.drain(),
                        &|_, sv| sv.val,
                        &|_| {},
                        pl,
                    )?;
                    Ok(func)
                },
                pl,
                (),
            )?;

            // -- Build lcp2bucket VFunc (sequential, small) --
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

            // Sequential: num_buckets is small and we avoid materializing the key set
            let lcp2bucket =
                <VFunc<BitPrefix, BitFieldVec<Box<[usize]>>, S1, E1>>::try_new_with_builder(
                    FromCloneableIntoIterator::new(
                        (0..num_buckets)
                            .map(|b| BitPrefix::new(&extended_first_keys[b], lcp_bit_lengths[b])),
                    ),
                    FromCloneableIntoIterator::new(0..num_buckets),
                    lcp2bucket_builder,
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

            Ok(result)
        }
    }
} // mod build

#[cfg(feature = "rayon")]
pub(crate) use build::{lcp_bits, lcp_bits_nul, log2_bucket_size};

/// A monotone minimal perfect hash function for sorted integer keys based
/// on longest common bit-prefixes (LCPs).
///
/// See the [module documentation] for the algorithmic description.
///
/// This structure implements the [`TryIntoUnaligned`] trait, allowing it to be
/// converted into (usually faster) structures using unaligned access.
///
/// # Type parameters
///
/// - `K`: the integer key type.
/// - `D`: the backing store for [`VFunc`] data (e.g.,
///   [`BitFieldVec`]).
/// - `S0`: the [signature type] for the key map
///   (`offset_lcp_length`).
/// - `E0`: the [`ShardEdge`] for the key map.
/// - `S1`: the  [signature type] for the prefix-to-bucket map
///   (`lcp2bucket`).
/// - `E1`: the [`ShardEdge`] for the prefix-to-bucket map.
///
/// # Examples
///
/// See [`try_new`].
///
/// [module documentation]: self
/// [signature type]: Sig
/// [`try_new`]: LcpMmphfInt::try_new
#[derive(Debug, Clone, MemSize, MemDbg)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LcpMmphfInt<
    K,
    D = BitFieldVec<Box<[usize]>>,
    S0 = [u64; 2],
    E0 = FuseLge3Shards,
    S1 = [u64; 1],
    E1 = Fuse3NoShards,
> {
    /// Number of keys.
    pub(crate) n: usize,
    /// Log2 of bucket size.
    pub(crate) log2_bucket_size: usize,
    /// Maps each key to `(lcp_bit_length << log2_bucket_size) | offset`.
    pub(crate) offset_lcp_length: VFunc<K, D, S0, E0>,
    /// Maps each LCP bit-prefix to its bucket index.
    pub(crate) lcp2bucket: VFunc<IntBitPrefix<K>, D, S1, E1>,
}

impl<
    K: PrimitiveInteger + ToSig<S0>,
    D: SliceByValue<Value = usize>,
    S0: Sig,
    E0: ShardEdge<S0, 3>,
    S1: Sig,
    E1: ShardEdge<S1, 3>,
> LcpMmphfInt<K, D, S0, E0, S1, E1>
where
    IntBitPrefix<K>: ToSig<S1>,
{
    /// Returns the rank (0-based position) of the given key in the
    /// original sorted sequence.
    ///
    /// If the key was not in the original set, the result is arbitrary
    /// (same contract as [`VFunc::get`]).
    #[inline]
    pub fn get(&self, key: K) -> usize {
        let packed = self.offset_lcp_length.get(key);
        let lcp_bit_length = packed >> self.log2_bucket_size;
        let offset = packed & ((1 << self.log2_bucket_size) - 1);
        // XOR with K::MIN maps signed numeric order to bit-lexicographic
        // order by flipping the sign bit; for unsigned types K::MIN is 0,
        // so this is a no-op.
        let prefix = IntBitPrefix::new(key ^ K::MIN, lcp_bit_length);
        let bucket = self.lcp2bucket.get(prefix);
        (bucket << self.log2_bucket_size) + offset
    }
}

impl<
    K: PrimitiveInteger,
    D: SliceByValue<Value = usize>,
    S0: Sig,
    E0: ShardEdge<S0, 3>,
    S1: Sig,
    E1: ShardEdge<S1, 3>,
> LcpMmphfInt<K, D, S0, E0, S1, E1>
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
#[derive(Debug, Clone, MemSize, MemDbg)]
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
///
/// [`Xxh3`]: xxh3::Xxh3
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

/// Computes a signature from raw bytes and a bit length,
/// matching the [`BitPrefix`] [`ToSig`] implementation but
/// without allocating a `BitPrefix`.
#[inline]
pub(crate) fn bit_prefix_sig<S: Sig>(bytes: &[u8], bit_length: usize, seed: u64) -> S {
    let mut hasher = xxh3::Xxh3::with_seed(seed);
    hash_bit_prefix_raw(&mut hasher, bytes, bit_length);
    S::from_hasher(&hasher)
}

impl ToSig<[u64; 1]> for BitPrefix {
    #[inline]
    fn to_sig(key: impl Borrow<Self>, seed: u64) -> [u64; 1] {
        let bp = key.borrow();
        bit_prefix_sig(&bp.bytes, bp.bit_length, seed)
    }
}

/// A monotone minimal perfect hash function for sorted byte-sequence keys based
/// on longest common prefixes (LCPs).
///
/// See the [module documentation] for the algorithmic description.
/// See [`LcpMmphfStr`] and [`LcpMmphfSliceU8`] for common instantiations,
/// and [`LcpMmphfInt`] for integer keys.
///
/// This structure implements the [`TryIntoUnaligned`] trait, allowing it to be
/// converted into (usually faster) structures using unaligned access.
///
/// # Type parameters
///
/// - `K`: the integer key type.
/// - `D`: the backing store for [`VFunc`] data (e.g.,
///   [`BitFieldVec`]).
/// - `S0`: the [signature type] for the key map
///   (`offset_lcp_length`).
/// - `E0`: the [`ShardEdge`] for the key map.
/// - `S1`: the  [signature type] for the prefix-to-bucket map
///   (`lcp2bucket`).
/// - `E1`: the [`ShardEdge`] for the prefix-to-bucket map.
///
/// # Examples
///
/// See [`try_new`]. See also [`LcpMmphfStr`] and [`LcpMmphfSliceU8`]
/// for common instantiations.
///
/// [module documentation]: self
/// [signature type]: Sig
/// [`try_new`]: LcpMmphf::try_new
#[derive(Debug, Clone, MemSize, MemDbg)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LcpMmphf<
    K: ?Sized,
    D = BitFieldVec<Box<[usize]>>,
    S0 = [u64; 2],
    E0 = FuseLge3Shards,
    S1 = [u64; 1],
    E1 = Fuse3NoShards,
> {
    /// Number of keys.
    pub(crate) n: usize,
    /// Log2 of bucket size.
    pub(crate) log2_bucket_size: usize,
    /// Maps each key to `(lcp_bit_length << log2_bucket_size) | offset`.
    pub(crate) offset_lcp_length: VFunc<K, D, S0, E0>,
    /// Maps each LCP bit-prefix to its bucket index.
    pub(crate) lcp2bucket: VFunc<BitPrefix, D, S1, E1>,
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
/// # use sux::traits::TryIntoUnaligned;
/// # use sux::utils::FromSlice;
/// let keys = vec![
///     "alpha".to_owned(),
///     "beta".to_owned(),
///     "delta".to_owned(),
///     "gamma".to_owned(),
/// ];
///
/// let func =
///     <LcpMmphfStr>::try_new(FromSlice::new(&keys), keys.len(), no_logging![])?.try_into_unaligned()?;
///
/// for (i, key) in keys.iter().enumerate() {
///     assert_eq!(func.get(key.as_str()), i);
/// }
/// # Ok(())
/// # }
/// # #[cfg(not(feature = "rayon"))]
/// # fn main() {}
/// ```
pub type LcpMmphfStr<
    D = BitFieldVec<Box<[usize]>>,
    S0 = [u64; 2],
    E0 = FuseLge3Shards,
    S1 = [u64; 1],
    E1 = Fuse3NoShards,
> = LcpMmphf<str, D, S0, E0, S1, E1>;

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
/// # use sux::traits::TryIntoUnaligned;
/// # use sux::utils::FromSlice;
/// let keys: Vec<Vec<u8>> = vec![
///     b"alpha".to_vec(),
///     b"beta".to_vec(),
///     b"delta".to_vec(),
///     b"gamma".to_vec(),
/// ];
///
/// let func = <LcpMmphfSliceU8>::try_new(
///     FromSlice::new(&keys),
///     keys.len(),
///     no_logging![],
/// )?.try_into_unaligned()?;
///
/// for (i, key) in keys.iter().enumerate() {
///     assert_eq!(func.get(key.as_slice()), i);
/// }
/// # Ok(())
/// # }
/// # #[cfg(not(feature = "rayon"))]
/// # fn main() {}
/// ```
pub type LcpMmphfSliceU8<
    D = BitFieldVec<Box<[usize]>>,
    S0 = [u64; 2],
    E0 = FuseLge3Shards,
    S1 = [u64; 1],
    E1 = Fuse3NoShards,
> = LcpMmphf<[u8], D, S0, E0, S1, E1>;

impl<
    K: ?Sized + AsRef<[u8]> + ToSig<S0>,
    D: SliceByValue<Value = usize>,
    S0: Sig,
    E0: ShardEdge<S0, 3>,
    S1: Sig,
    E1: ShardEdge<S1, 3>,
> LcpMmphf<K, D, S0, E0, S1, E1>
where
    BitPrefix: ToSig<S1>,
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
        // Compute the lcp2bucket signature directly from the key bytes
        // without allocating a BitPrefix.
        let key_bytes: &[u8] = key.as_ref();
        let seed = self.lcp2bucket.seed;
        let sig: S1 = if lcp_bit_length <= key_bytes.len() * 8 {
            bit_prefix_sig(key_bytes, lcp_bit_length, seed)
        } else {
            // Rare: LCP extends into the virtual NUL (at most 8 extra bits).
            // Since the NUL byte is 0x00, masking is a no-op, so we can
            // just hash all key bytes + the NUL + the bit length.
            let mut hasher = xxh3::Xxh3::with_seed(seed);
            hasher.update(key_bytes);
            hasher.update(&[0u8]);
            hasher.update(&lcp_bit_length.to_ne_bytes());
            S1::from_hasher(&hasher)
        };
        let bucket = self.lcp2bucket.get_by_sig(sig);
        (bucket << self.log2_bucket_size) + offset
    }
}

impl<K: ?Sized, D: SliceByValue, S0: Sig, E0: ShardEdge<S0, 3>, S1: Sig, E1: ShardEdge<S1, 3>>
    LcpMmphf<K, D, S0, E0, S1, E1>
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

// ── Aligned ↔ Unaligned conversions ──────────────────────────────────

// -- LcpMmphfInt --

impl<K, S0: Sig, E0: ShardEdge<S0, 3>, S1: Sig, E1: ShardEdge<S1, 3>>
    From<Unaligned<LcpMmphfInt<K, BitFieldVec<Box<[usize]>>, S0, E0, S1, E1>>>
    for LcpMmphfInt<K, BitFieldVec<Box<[usize]>>, S0, E0, S1, E1>
{
    fn from(f: Unaligned<LcpMmphfInt<K, BitFieldVec<Box<[usize]>>, S0, E0, S1, E1>>) -> Self {
        LcpMmphfInt {
            n: f.n,
            log2_bucket_size: f.log2_bucket_size,
            offset_lcp_length: f.offset_lcp_length.into(),
            lcp2bucket: f.lcp2bucket.into(),
        }
    }
}

impl<K, S0: Sig, E0: ShardEdge<S0, 3>, S1: Sig, E1: ShardEdge<S1, 3>> TryIntoUnaligned
    for LcpMmphfInt<K, BitFieldVec<Box<[usize]>>, S0, E0, S1, E1>
{
    type Unaligned = LcpMmphfInt<K, Unaligned<BitFieldVec<Box<[usize]>>>, S0, E0, S1, E1>;
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

impl<K: ?Sized, S0: Sig, E0: ShardEdge<S0, 3>, S1: Sig, E1: ShardEdge<S1, 3>>
    From<Unaligned<LcpMmphf<K, BitFieldVec<Box<[usize]>>, S0, E0, S1, E1>>>
    for LcpMmphf<K, BitFieldVec<Box<[usize]>>, S0, E0, S1, E1>
{
    fn from(f: Unaligned<LcpMmphf<K, BitFieldVec<Box<[usize]>>, S0, E0, S1, E1>>) -> Self {
        LcpMmphf {
            n: f.n,
            log2_bucket_size: f.log2_bucket_size,
            offset_lcp_length: f.offset_lcp_length.into(),
            lcp2bucket: f.lcp2bucket.into(),
        }
    }
}

impl<K: ?Sized, S0: Sig, E0: ShardEdge<S0, 3>, S1: Sig, E1: ShardEdge<S1, 3>> TryIntoUnaligned
    for LcpMmphf<K, BitFieldVec<Box<[usize]>>, S0, E0, S1, E1>
{
    type Unaligned = LcpMmphf<K, Unaligned<BitFieldVec<Box<[usize]>>>, S0, E0, S1, E1>;
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
