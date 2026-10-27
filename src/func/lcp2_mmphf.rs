/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

#![allow(clippy::type_complexity)]

//! Two-step LCP-based monotone minimal perfect hash functions.
//!
//! This module contains structures analogous to those in
//! [`lcp_mmphf`]; however, they use a secondary [`VFunc`] for
//! infrequent prefix lengths, similarly to a [`VFunc2`], providing
//! some space savings at the cost of slightly slower queries.
//!
//! [`Lcp2MmphfInt`] works with any primitive integer type, whereas
//! [`Lcp2Mmphf`] works with any byte-sequence key type (`K: AsRef<[u8]>`). Type
//! aliases [`Lcp2MmphfStr`] and [`Lcp2MmphfSliceU8`] are provided for
//! convenience. For the byte-sequence variant, keys must not contain zeros, as
//! a virtual zero byte is appended internally to ensure prefix-freeness.
//! Alternatively, they must be prefix-free, in which case they can contain
//! zeros.
//!
//! These structures implement the [`TryIntoUnaligned`] trait, allowing them to
//! be converted into (usually faster) structures using unaligned access.
//!
//! Like those in [`lcp_mmphf`], these structures use ε-cost sharding, as
//! described by Sebastiano Vigna in “[ε–Cost Sharding: Scaling Hypergraph–Based
//! Data Structures to Trillions of Keys]”, 2026.
//!
//! # References
//!
//! Djamal Belazzougui, Paolo Boldi, Rasmus Pagh, and Sebastiano Vigna.
//! [Theory and practice of monotone minimal perfect hashing]. *ACM Journal of
//! Experimental Algorithmics*, 16(3):3.2:1−3.2:26, 2011.
//!
//! [`lcp_mmphf`]: super::lcp_mmphf
//! [`VFunc2`]: crate::func::VFunc2
//! [Theory and practice of monotone minimal perfect hashing]: https://doi.org/10.1145/1963190.2025378
//! [ε–Cost Sharding: Scaling Hypergraph–Based Data Structures to Trillions of Keys]: https://arxiv.org/abs/2503.18397

use crate::bits::BitFieldVec;
use crate::func::VFunc;
use crate::func::lcp_mmphf::{BitPrefix, IntBitPrefix, bit_prefix_sig};
use crate::func::shard_edge::{Fuse3Shards, FuseAug3Shards, ShardEdge};
use crate::utils::*;
use mem_dbg::*;
use num_primitive::PrimitiveInteger;
use value_traits::slices::SliceByValue;
use xxhash_rust::xxh3;

/// A two-step monotone minimal perfect hash function for sorted integer keys based
/// on longest common bit-prefixes (LCPs).
///
/// See the [module documentation] for the algorithmic description.
///
/// Instances of this structure are immutable; they are built using [`try_new`]
/// or one of its variants, and can be serialized using [ε-serde] or [`serde`].
/// The construction can be configured by passing a [`VBuilder`] to the
/// `*_with_builder` constructors. Since signatures are computed from the
/// native representation of keys (see [`ToSig`]), a serialized function
/// must be queried on a machine with the same endianness.
///
/// This structure implements the [`TryIntoUnaligned`] trait, allowing it to be
/// converted into (usually faster) structures using unaligned access.
///
/// The `lcp_mmphf` binary (`cli` feature) builds and serializes instances of
/// this structure when invoked with the `--two-step` option.
///
/// # Alternatives
///
/// - [`LcpMmphfInt`] has slightly faster queries, but uses more space.
///
/// - [`PHastR`] is faster and uses less space if any bijection between the
///   keys and [0 . . *n*) is sufficient.
///
/// - A [`SignedFunc`] wrapping this structure rejects (with high
///   probability) keys outside the original set.
///
/// # Type parameters
///
/// - `K`: the integer key type; it must implement [`ToSig`] (see the
///   [`VFunc`] documentation).
/// - `D`: the backing store for [`VFunc`] data (e.g.,
///   [`BitFieldVec`]).
/// - `S0`: the [signature type] for the key maps (`lcp_freq_len_offset`
///   and `lcp_infreq_len`).
/// - `E0`: the [`ShardEdge`] for the key map `lcp_freq_len_offset`.
/// - `F0`: the [`ShardEdge`] for `lcp_infreq_len`. Defaults to
///   [`Fuse3Shards`].
/// - `S1`: the [signature type] for the prefix-to-bucket map
///   (`lcp_to_bucket`).
/// - `E1`: the [`ShardEdge`] for the prefix-to-bucket map.
///
/// # Examples
///
/// See [`try_new`].
///
/// [module documentation]: self
/// [signature type]: Sig
/// [`try_new`]: Lcp2MmphfInt::try_new
/// [ε-serde]: https://crates.io/crates/epserde
/// [`serde`]: https://crates.io/crates/serde
/// [`VBuilder`]: crate::func::VBuilder
/// [`LcpMmphfInt`]: crate::func::LcpMmphfInt
/// [`PHastR`]: crate::func::PHastR
/// [`SignedFunc`]: crate::func::SignedFunc
#[doc(alias("monotone", "mmphf"))]
#[derive(Clone, MemSize, MemDbg)]
#[cfg_attr(
    feature = "epserde",
    derive(epserde::Epserde),
    epserde(phantom(K, S0, S1))
)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "VFunc<K, D, S0, E0>: serde::Serialize, VFunc<K, D, S0, F0>: serde::Serialize, VFunc<IntBitPrefix<K>, D, S1, E1>: serde::Serialize",
        deserialize = "VFunc<K, D, S0, E0>: serde::Deserialize<'de>, VFunc<K, D, S0, F0>: serde::Deserialize<'de>, VFunc<IntBitPrefix<K>, D, S1, E1>: serde::Deserialize<'de>"
    ))
)]
pub struct Lcp2MmphfInt<
    K,
    D = BitFieldVec<Box<[usize]>>,
    S0 = [u64; 2],
    E0 = FuseAug3Shards,
    F0 = Fuse3Shards,
    S1 = [u64; 2],
    E1 = FuseAug3Shards,
> {
    /// The number of keys.
    pub(crate) n: usize,
    /// The base-2 logarithm of the bucket size.
    pub(crate) log2_bucket_size: usize,
    /// Maps key → lcp_freq_len | offset.
    pub(crate) lcp_freq_len_offset: VFunc<K, D, S0, E0>,
    /// Maps escaped keys to their full LCP bit length.
    pub(crate) lcp_infreq_len: VFunc<K, D, S0, F0>,
    /// Maps frequent LCP indices back to actual LCP bit lengths.
    pub(crate) remap: Box<[usize]>,
    /// Escape sentinel for the frequent LCP part (2^r − 1).
    pub(crate) escape: usize,
    /// Maps each LCP bit-prefix to its bucket index.
    pub(crate) lcp_to_bucket: VFunc<IntBitPrefix<K>, D, S1, E1>,
}

impl<K, D, S0, E0, F0, S1, E1> Lcp2MmphfInt<K, D, S0, E0, F0, S1, E1> {
    /// Returns the number of keys.
    pub const fn len(&self) -> usize {
        self.n
    }
    /// Returns `true` if the function contains no keys.
    pub const fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Returns the seed used to compute the signatures of the keys.
    pub const fn seed(&self) -> u64 {
        self.lcp_freq_len_offset.seed
    }
}

impl<K: PrimitiveInteger, D: SliceByValue, S0, E0, F0, S1, E1> std::fmt::Debug
    for Lcp2MmphfInt<K, D, S0, E0, F0, S1, E1>
where
    VFunc<K, D, S0, E0>: std::fmt::Debug,
    VFunc<K, D, S0, F0>: std::fmt::Debug,
    VFunc<IntBitPrefix<K>, D, S1, E1>: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Lcp2MmphfInt")
            .field("n", &self.n)
            .field("log2_bucket_size", &self.log2_bucket_size)
            .field("lcp_freq_len_offset", &self.lcp_freq_len_offset)
            .field("lcp_infreq_len", &self.lcp_infreq_len)
            .field("remap", &self.remap)
            .field("escape", &self.escape)
            .field("lcp_to_bucket", &self.lcp_to_bucket)
            .finish()
    }
}

impl<
    K: PrimitiveInteger + ToSig<S0> + Copy,
    D: SliceByValue<Value = usize>,
    S0: Sig,
    E0: ShardEdge<S0, 3>,
    F0: ShardEdge<S0, 3>,
    S1: Sig,
    E1: ShardEdge<S1, 3>,
> Lcp2MmphfInt<K, D, S0, E0, F0, S1, E1>
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
        let sig = K::to_sig(key, self.lcp_freq_len_offset.seed);
        self.get_by_sig(key, sig)
    }

    #[inline]
    pub(crate) fn get_by_sig(&self, key: K, sig: S0) -> usize {
        let packed = self.lcp_freq_len_offset.get_by_sig(sig);
        let offset = packed & ((1 << self.log2_bucket_size) - 1);
        let frequent_lcp = packed >> self.log2_bucket_size;
        // Absent keys yield an arbitrary packed value; clamp the decoded LCP
        // length to the key width so IntBitPrefix::new cannot overflow. Present
        // keys always have lcp_bit_len <= K::BITS, so this is a no-op for them.
        let lcp_bit_len = if frequent_lcp != self.escape {
            self.remap[frequent_lcp]
        } else {
            self.lcp_infreq_len.get_by_sig(sig)
        }
        .min(K::BITS as usize);
        let prefix = IntBitPrefix::new(key ^ K::MIN, lcp_bit_len);
        let bucket = self.lcp_to_bucket.get(prefix);
        (bucket << self.log2_bucket_size) + offset
    }
}

#[cfg(feature = "rayon")]
mod build {
    use super::*;
    use crate::func::VBuilder;
    use crate::func::lcp_mmphf::{
        BytesLcpScan, IntLcpScan, bucket_layout, build_bytes_lcp_to_bucket,
        build_int_lcp_to_bucket, log_bits_per_key,
    };
    use crate::func::vfunc2::{HybridMap, Remap};
    use anyhow::Result;
    use dsi_progress_logger::ProgressLog;
    use lender::*;
    use rdst::RadixKey;
    use std::borrow::Borrow;
    use std::ops::{BitXor, BitXorAssign};

    // Always u32 so that LCP bit-lengths above 65535 do not silently wrap on
    // 32-bit targets (where this alias was previously u16).
    type LcpLen = u32;

    /// Returns the number of keys whose bucket has a given LCP bit length,
    /// and the maximum LCP bit length.
    fn lcp_counts(
        lcp_bit_lens: &[LcpLen],
        log2_bs: usize,
        n: usize,
    ) -> (HybridMap<usize, usize>, usize) {
        let bucket_size = 1 << log2_bs;
        let mut counts = HybridMap::new(None, 0);
        let mut max_lcp = 0;
        for (b, &lcp) in lcp_bit_lens.iter().enumerate() {
            let lcp = lcp as usize;
            max_lcp = max_lcp.max(lcp);
            // All buckets are full, except possibly for the last one
            counts.add(lcp, bucket_size.min(n - b * bucket_size));
        }
        (counts, max_lcp)
    }

    /// Pushes the signatures of the keys of a bucket, associating with each
    /// key the LCP bit length of the bucket packed with the offset of the key
    /// in the bucket, and empties `sigs`.
    fn push_bucket<S: Sig>(
        sigs: &mut Vec<S>,
        lcp: usize,
        log2_bs: usize,
        max_value: &mut u64,
        push: &mut dyn FnMut(SigVal<S, u64>) -> Result<()>,
    ) -> Result<()> {
        for (i, &sig) in sigs.iter().enumerate() {
            let packed = ((lcp as u64) << log2_bs) | i as u64;
            *max_value = (*max_value).max(packed);
            push(SigVal { sig, val: packed })?;
        }
        sigs.clear();
        Ok(())
    }

    /// Builds, from a store associating with each key its packed LCP bit
    /// length and offset, the two functions of a two-step LCP-based
    /// MMPHF: the first maps keys to their packed frequent LCP bit length and
    /// offset, the second maps keys with an infrequent LCP bit length to
    /// their LCP bit length.
    ///
    /// Returns the two functions and the remapping of the frequent LCP bit
    /// lengths.
    fn build_key_funcs<K, S0, E0, F0>(
        builder: &mut VBuilder<BitFieldVec<Box<[usize]>>, S0, E0>,
        seed: u64,
        store: &mut (impl ShardStore<S0, u64> + ?Sized),
        lcp_bit_lens: &[LcpLen],
        log2_bs: usize,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> Result<(
        VFunc<K, BitFieldVec<Box<[usize]>>, S0, E0>,
        VFunc<K, BitFieldVec<Box<[usize]>>, S0, F0>,
        Remap<usize>,
    )>
    where
        K: ?Sized + ToSig<S0>,
        S0: Sig + Send + Sync,
        E0: ShardEdge<S0, 3> + MemSize + mem_dbg::FlatType,
        F0: ShardEdge<S0, 3> + MemSize + mem_dbg::FlatType,
        SigVal<S0, usize>: RadixKey,
        SigVal<S0, u64>: RadixKey,
        SigVal<E0::LocalSig, usize>: BitXor + BitXorAssign,
        SigVal<E0::LocalSig, u64>: BitXor + BitXorAssign,
        SigVal<F0::LocalSig, usize>: BitXor + BitXorAssign,
        SigVal<F0::LocalSig, u64>: BitXor + BitXorAssign,
    {
        // -- Compute the remapping from LCP frequencies --
        let (counts, max_lcp) = lcp_counts(lcp_bit_lens, log2_bs, store.len());
        let remap = Remap::new(&counts, store.len(), max_lcp, pl);

        let bucket_mask = (1 << log2_bs) - 1;
        let lcp = |v: u64| (v >> log2_bs) as usize;
        let escaped = |v: u64| remap.inv_map.get(lcp(v)) == remap.escape;

        // -- Build lcp_freq_len_offset --
        pl.push_log_target(" ▸ lcp_freq_len_offset");
        pl.info(format_args!(
            "Building key → lcp_freq_len | offset function..."
        ));
        let shard_edge = builder.shard_edge;
        let result = builder.try_build_first_func::<K, u64>(
            seed,
            shard_edge,
            (remap.escape << log2_bs) | bucket_mask,
            store,
            &|v| (remap.inv_map.get(lcp(v)) << log2_bs) | (v as usize & bucket_mask),
            &escaped,
            pl,
        );
        // Pop before propagating errors, or retries would nest log targets
        pl.pop_log_target();
        let (lcp_freq_len_offset, escaped_counts) = result?;

        // -- Build lcp_infreq_len (escaped keys only) --
        let lcp_infreq_len = if remap.num_escaped > 0 {
            pl.push_log_target(" ▸ lcp_infreq_len");
            pl.info(format_args!(
                "Building escaped key → lcp_infreq_len function..."
            ));
            let result = builder.try_build_escaped_func::<K, u64, F0>(
                seed,
                store,
                &escaped_counts,
                &escaped,
                max_lcp,
                &lcp,
                pl,
            );
            pl.pop_log_target();
            result?
        } else {
            VFunc::empty()
        };

        Ok((lcp_freq_len_offset, lcp_infreq_len, remap))
    }

    impl<
        K: PrimitiveInteger + ToSig<S0> + std::fmt::Debug + Send + Sync + Copy + Ord,
        S0: Sig + Send + Sync,
        E0: ShardEdge<S0, 3> + MemSize + mem_dbg::FlatType,
        F0: ShardEdge<S0, 3> + MemSize + mem_dbg::FlatType,
        S1: Sig + Send + Sync,
        E1: ShardEdge<S1, 3> + MemSize + mem_dbg::FlatType,
    > Lcp2MmphfInt<K, BitFieldVec<Box<[usize]>>, S0, E0, F0, S1, E1>
    where
        IntBitPrefix<K>: ToSig<S1>,
        SigVal<S0, usize>: RadixKey,
        SigVal<S0, u64>: RadixKey,
        SigVal<E0::LocalSig, usize>: BitXor + BitXorAssign,
        SigVal<E0::LocalSig, u64>: BitXor + BitXorAssign,
        SigVal<F0::LocalSig, usize>: BitXor + BitXorAssign,
        SigVal<F0::LocalSig, u64>: BitXor + BitXorAssign,
        SigVal<S1, usize>: RadixKey,
        SigVal<E1::LocalSig, usize>: BitXor + BitXorAssign,
    {
        /// Creates a two-step LCP-based MMPHF for integers using default
        /// [`VBuilder`] settings.
        ///
        /// This is a convenience wrapper around
        /// [`try_new_with_builder`]. Use that method if you need to
        /// configure construction parameters such as offline mode, thread
        /// count, or sharding overhead.
        ///
        /// If keys are available as a slice, [`try_par_new`] parallelizes
        /// the hash computation for faster construction.
        ///
        /// The keys must be provided in strictly increasing order.
        ///
        /// Keys must be provided as a [`FallibleRewindableLender`]. The [`lenders`]
        /// module provides easy ways to build such lenders. The argument `n`
        /// must be the number of keys returned by the lender: a mismatch
        /// causes an error.
        ///
        /// # Examples
        ///
        /// ```rust
        /// # #[cfg(feature = "rayon")]
        /// # fn main() -> anyhow::Result<()> {
        /// # use sux::func::Lcp2MmphfInt;
        /// # use sux::traits::TryIntoUnaligned;
        /// # use dsi_progress_logger::no_logging;
        /// # use sux::utils::FromSlice;
        /// let keys: Vec<u64> = vec![10, 20, 30, 40, 50];
        /// let func =
        ///     Lcp2MmphfInt::<u64>::try_new(FromSlice::new(&keys), keys.len(), no_logging![])?.try_into_unaligned()?;
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

        /// Creates a two-step LCP-based MMPHF for integers using the
        /// given [`VBuilder`] configuration.
        ///
        /// The builder controls construction parameters such as [offline
        /// mode], [thread count], [sharding overhead], and [PRNG seed].
        ///
        /// See also [`try_par_new_with_builder`] for parallel hash
        /// computation from slices.
        ///
        /// The keys must be provided in strictly increasing order.
        ///
        /// Keys must be provided as a [`FallibleRewindableLender`]. The [`lenders`]
        /// module provides easy ways to build such lenders. The argument `n`
        /// must be the number of keys returned by the lender: a mismatch
        /// causes an error.
        ///
        /// # Examples
        ///
        /// ```rust
        /// # #[cfg(feature = "rayon")]
        /// # fn main() -> anyhow::Result<()> {
        /// # use sux::func::{Lcp2MmphfInt, VBuilder};
        /// # use sux::traits::TryIntoUnaligned;
        /// # use dsi_progress_logger::no_logging;
        /// # use sux::utils::FromSlice;
        /// let keys: Vec<u64> = vec![10, 20, 30, 40, 50];
        /// let func = Lcp2MmphfInt::<u64>::try_new_with_builder(
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

        /// Internal constructor returning both the MMPHF and the keys lender.
        pub(crate) fn try_new_inner<
            P: ProgressLog + Clone + Send + Sync,
            L: FallibleRewindableLender<
                    RewindError: std::error::Error + Send + Sync + 'static,
                    Error: std::error::Error + Send + Sync + 'static,
                > + for<'lend> FallibleLending<'lend, Lend = &'lend K>,
        >(
            mut keys: L,
            n: usize,
            builder: VBuilder<BitFieldVec<Box<[usize]>>, S0, E0>,
            pl: &mut P,
        ) -> Result<(Self, L)> {
            if n == 0 {
                anyhow::ensure!(keys.next()?.is_none(), "Expected 0 keys but got at least 1");
                return Ok((Self::empty(), keys));
            }

            let (log2_bs, num_buckets) = bucket_layout(n, pl);
            let mut builder = builder.expected_num_keys(n);

            builder.try_with_retries(
                keys,
                |builder, seed, keys, pl| {
                    let mut populate =
                        |seed: u64,
                         push: &mut dyn FnMut(SigVal<S0, u64>) -> Result<()>,
                         pl: &mut P,
                         scan: &mut IntLcpScan<K, LcpLen>| {
                            // Signatures of the keys of the current bucket
                            let mut sigs: Vec<S0> = Vec::with_capacity(1 << log2_bs);
                            let mut max_value = 0;
                            while let Some(&key) = keys.next()? {
                                pl.light_update();
                                // Start of a new bucket; flush the previous one.
                                if let Some(lcp) = scan.push(key)? {
                                    push_bucket(&mut sigs, lcp, log2_bs, &mut max_value, push)?;
                                }
                                sigs.push(K::to_sig(key, seed));
                            }

                            // Flush the last (possibly partial) bucket.
                            if !sigs.is_empty() {
                                let lcp = scan.finish()?;
                                push_bucket(&mut sigs, lcp, log2_bs, &mut max_value, push)?;
                            }

                            anyhow::ensure!(
                                scan.num_keys() == n,
                                "Expected {n} keys but got {}",
                                scan.num_keys()
                            );
                            assert_eq!(scan.lcp_bit_lens.len(), num_buckets);

                            Ok(max_value)
                        };

                    builder.try_solve_once(
                        seed,
                        &mut populate,
                        &mut |builder,
                              seed,
                              mut store,
                              _max_value,
                              _num_keys,
                              pl: &mut P,
                              scan: &mut IntLcpScan<K, LcpLen>| {
                            Self::build_from_store(builder, seed, &mut *store, scan, log2_bs, pl)
                        },
                        pl,
                        &mut IntLcpScan::new(log2_bs, num_buckets),
                    )
                },
                |keys| Ok(keys.rewind()?),
                pl,
            )
        }

        /// Returns a function with no keys.
        fn empty() -> Self {
            Self {
                n: 0,
                log2_bucket_size: 0,
                lcp_freq_len_offset: VFunc::empty(),
                lcp_infreq_len: VFunc::empty(),
                remap: Box::new([]),
                escape: 0,
                lcp_to_bucket: VFunc::empty(),
            }
        }

        /// Builds a function from a store associating with each key its
        /// packed LCP bit length and offset.
        fn build_from_store(
            builder: &mut VBuilder<BitFieldVec<Box<[usize]>>, S0, E0>,
            seed: u64,
            store: &mut (impl ShardStore<S0, u64> + ?Sized),
            scan: &IntLcpScan<K, LcpLen>,
            log2_bs: usize,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self> {
            let n = scan.num_keys();
            let (lcp_freq_len_offset, lcp_infreq_len, remap) =
                build_key_funcs(builder, seed, store, &scan.lcp_bit_lens, log2_bs, pl)?;

            let lcp_to_bucket = build_int_lcp_to_bucket(
                &scan.bucket_first_keys,
                &scan.lcp_bit_lens,
                VBuilder::default().set_from(builder),
                pl,
            )?;

            let result = Self {
                n,
                log2_bucket_size: log2_bs,
                lcp_freq_len_offset,
                lcp_infreq_len,
                remap: remap.remap,
                escape: remap.escape,
                lcp_to_bucket,
            };
            log_bits_per_key(pl, &result, n);
            Ok(result)
        }

        /// Creates a two-step LCP-based monotone minimal perfect hash function for
        /// integers from a slice, using parallel hash computation and default
        /// [`VBuilder`] settings.
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
        /// # use sux::func::Lcp2MmphfInt;
        /// # use sux::traits::TryIntoUnaligned;
        /// # use dsi_progress_logger::no_logging;
        /// let keys: Vec<u64> = vec![10, 20, 30, 40, 50];
        /// let func =
        ///     Lcp2MmphfInt::<u64>::try_par_new(&keys, no_logging![])?.try_into_unaligned()?;
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

        /// Creates a two-step LCP-based monotone minimal perfect hash function for
        /// integers from a slice, using parallel hash computation and the given
        /// [`VBuilder`] configuration.
        ///
        /// This is the parallel counterpart of
        /// [`try_new_with_builder`].
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
        /// # use sux::func::{Lcp2MmphfInt, VBuilder};
        /// # use sux::traits::TryIntoUnaligned;
        /// # use dsi_progress_logger::no_logging;
        /// let keys: Vec<u64> = vec![10, 20, 30, 40, 50];
        /// let func = Lcp2MmphfInt::<u64>::try_par_new_with_builder(
        ///     &keys,
        ///     VBuilder::default(),
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
        pub fn try_par_new_with_builder<P: ProgressLog + Clone + Send + Sync>(
            keys: &[K],
            builder: VBuilder<BitFieldVec<Box<[usize]>>, S0, E0>,
            pl: &mut P,
        ) -> Result<Self> {
            let n = keys.len();
            if n == 0 {
                return Ok(Self::empty());
            }

            let (log2_bs, num_buckets) = bucket_layout(n, pl);

            // -- Sequential pass: compute bit-level LCPs --

            let mut scan = IntLcpScan::<K, LcpLen>::new(log2_bs, num_buckets);
            for &key in keys {
                scan.push(key)?;
            }
            scan.finish()?;

            let lcp_bit_lens = &scan.lcp_bit_lens;
            let bucket_mask = (1 << log2_bs) - 1;
            builder.expected_num_keys(n).try_par_populate_and_build(
                keys,
                &|i| ((lcp_bit_lens[i >> log2_bs] as u64) << log2_bs) | (i & bucket_mask) as u64,
                &mut |builder,
                      seed,
                      mut store,
                      _max_value,
                      _num_keys,
                      pl: &mut P,
                      _state: &mut ()| {
                    Self::build_from_store(builder, seed, &mut *store, &scan, log2_bs, pl)
                },
                pl,
                (),
            )
        }
    }

    impl<
        K: ?Sized + AsRef<[u8]> + ToSig<S0> + std::fmt::Debug,
        S0: Sig + Send + Sync,
        E0: ShardEdge<S0, 3> + MemSize + mem_dbg::FlatType,
        F0: ShardEdge<S0, 3> + MemSize + mem_dbg::FlatType,
        S1: Sig + Send + Sync,
        E1: ShardEdge<S1, 3> + MemSize + mem_dbg::FlatType,
    > Lcp2Mmphf<K, BitFieldVec<Box<[usize]>>, S0, E0, F0, S1, E1>
    where
        BitPrefix: ToSig<S1>,
        SigVal<S0, usize>: RadixKey,
        SigVal<S0, u64>: RadixKey,
        SigVal<E0::LocalSig, usize>: BitXor + BitXorAssign,
        SigVal<E0::LocalSig, u64>: BitXor + BitXorAssign,
        SigVal<F0::LocalSig, usize>: BitXor + BitXorAssign,
        SigVal<F0::LocalSig, u64>: BitXor + BitXorAssign,
        SigVal<S1, usize>: RadixKey,
        SigVal<E1::LocalSig, usize>: BitXor + BitXorAssign,
    {
        /// Creates a two-step LCP-based monotone minimal perfect hash function for
        /// byte-sequence keys using default [`VBuilder`] settings.
        ///
        /// This is a convenience wrapper around
        /// [`try_new_with_builder`]. Use that method if you need to
        /// configure construction parameters such as offline mode, thread
        /// count, or sharding overhead.
        ///
        /// If keys are available as a slice, [`try_par_new`] parallelizes
        /// the hash computation for faster construction.
        ///
        /// The keys must be in strictly increasing lexicographic order.
        /// The lender may yield references to any type `B` that borrows
        /// as `K` (e.g., `&String` for `K = str`, `&Vec<u8>` for
        /// `K = [u8]`).
        ///
        /// Keys must be provided as a [`FallibleRewindableLender`]. The [`lenders`]
        /// module provides easy ways to build such lenders. The argument `n`
        /// must be the number of keys returned by the lender: a mismatch
        /// causes an error.
        ///
        /// # Examples
        ///
        /// ```rust
        /// # #[cfg(feature = "rayon")]
        /// # fn main() -> anyhow::Result<()> {
        /// # use sux::func::Lcp2MmphfStr;
        /// # use sux::traits::TryIntoUnaligned;
        /// # use dsi_progress_logger::no_logging;
        /// # use sux::utils::FromSlice;
        /// let keys = vec!["a", "b", "c", "d", "e"];
        /// let func =
        ///     <Lcp2MmphfStr>::try_new(FromSlice::new(&keys), keys.len(), no_logging![])?.try_into_unaligned()?;
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

        /// Creates a two-step LCP-based monotone minimal perfect hash function for
        /// byte-sequence keys using the given [`VBuilder`] configuration.
        ///
        /// The builder controls construction parameters such as [offline
        /// mode], [thread count], [sharding overhead], and [PRNG seed].
        ///
        /// See also [`try_par_new_with_builder`] for parallel hash
        /// computation from slices.
        ///
        /// The keys must be in strictly increasing lexicographic order.
        ///
        /// Keys must be provided as a [`FallibleRewindableLender`]. The [`lenders`]
        /// module provides easy ways to build such lenders. The argument `n`
        /// must be the number of keys returned by the lender: a mismatch
        /// causes an error.
        ///
        /// # Examples
        ///
        /// ```rust
        /// # #[cfg(feature = "rayon")]
        /// # fn main() -> anyhow::Result<()> {
        /// # use sux::func::{Lcp2MmphfStr, VBuilder};
        /// # use sux::traits::TryIntoUnaligned;
        /// # use dsi_progress_logger::no_logging;
        /// # use sux::utils::FromSlice;
        /// let keys = vec!["a", "b", "c", "d", "e"];
        /// let func = <Lcp2MmphfStr>::try_new_with_builder(
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

        /// Internal constructor returning both the MMPHF and the keys lender.
        pub(crate) fn try_new_inner<
            B: ?Sized + AsRef<[u8]> + Borrow<K>,
            P: ProgressLog + Clone + Send + Sync,
            L: FallibleRewindableLender<
                    RewindError: std::error::Error + Send + Sync + 'static,
                    Error: std::error::Error + Send + Sync + 'static,
                > + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
        >(
            mut keys: L,
            n: usize,
            builder: VBuilder<BitFieldVec<Box<[usize]>>, S0, E0>,
            pl: &mut P,
        ) -> Result<(Self, L)> {
            if n == 0 {
                anyhow::ensure!(keys.next()?.is_none(), "Expected 0 keys but got at least 1");
                return Ok((Self::empty(), keys));
            }

            let (log2_bs, num_buckets) = bucket_layout(n, pl);
            let mut builder = builder.expected_num_keys(n);

            builder.try_with_retries(
                keys,
                |builder, seed, keys, pl| {
                    let mut populate =
                        |seed: u64,
                         push: &mut dyn FnMut(SigVal<S0, u64>) -> Result<()>,
                         pl: &mut P,
                         scan: &mut BytesLcpScan<LcpLen>| {
                            // Signatures of the keys of the current bucket
                            let mut sigs: Vec<S0> = Vec::with_capacity(1 << log2_bs);
                            let mut max_value = 0;
                            while let Some(key) = keys.next()? {
                                pl.light_update();
                                // Start of a new bucket; flush the previous one.
                                if let Some(lcp) = scan.push(key.as_ref())? {
                                    push_bucket(&mut sigs, lcp, log2_bs, &mut max_value, push)?;
                                }
                                sigs.push(K::to_sig(key.borrow(), seed));
                            }

                            // Flush the last (possibly partial) bucket.
                            if !sigs.is_empty() {
                                let lcp = scan.finish()?;
                                push_bucket(&mut sigs, lcp, log2_bs, &mut max_value, push)?;
                            }

                            anyhow::ensure!(
                                scan.num_keys() == n,
                                "Expected {n} keys but got {}",
                                scan.num_keys()
                            );
                            assert_eq!(scan.lcp_bit_lens.len(), num_buckets);

                            Ok(max_value)
                        };

                    builder.try_solve_once(
                        seed,
                        &mut populate,
                        &mut |builder,
                              seed,
                              mut store,
                              _max_value,
                              _num_keys,
                              pl: &mut P,
                              scan: &mut BytesLcpScan<LcpLen>| {
                            Self::build_from_store(builder, seed, &mut *store, scan, log2_bs, pl)
                        },
                        pl,
                        &mut BytesLcpScan::new(log2_bs, num_buckets),
                    )
                },
                |keys| Ok(keys.rewind()?),
                pl,
            )
        }

        /// Returns a function with no keys.
        fn empty() -> Self {
            Self {
                n: 0,
                log2_bucket_size: 0,
                lcp_freq_len_offset: VFunc::empty(),
                lcp_infreq_len: VFunc::empty(),
                remap: Box::new([]),
                escape: 0,
                lcp_to_bucket: VFunc::empty(),
            }
        }

        /// Builds a function from a store associating with each key its
        /// packed LCP bit length and offset.
        fn build_from_store(
            builder: &mut VBuilder<BitFieldVec<Box<[usize]>>, S0, E0>,
            seed: u64,
            store: &mut (impl ShardStore<S0, u64> + ?Sized),
            scan: &BytesLcpScan<LcpLen>,
            log2_bs: usize,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> Result<Self> {
            let n = scan.num_keys();
            let (lcp_freq_len_offset, lcp_infreq_len, remap) =
                build_key_funcs(builder, seed, store, &scan.lcp_bit_lens, log2_bs, pl)?;

            let lcp_to_bucket = build_bytes_lcp_to_bucket(
                &scan.bucket_first_keys,
                &scan.lcp_bit_lens,
                VBuilder::default().set_from(builder),
                pl,
            )?;

            let result = Self {
                n,
                log2_bucket_size: log2_bs,
                lcp_freq_len_offset,
                lcp_infreq_len,
                remap: remap.remap,
                escape: remap.escape,
                lcp_to_bucket,
            };
            log_bits_per_key(pl, &result, n);
            Ok(result)
        }

        /// Creates a two-step LCP-based monotone minimal perfect hash function for
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
        /// # use sux::func::Lcp2MmphfStr;
        /// # use sux::traits::TryIntoUnaligned;
        /// # use dsi_progress_logger::no_logging;
        /// let keys = vec!["a", "b", "c", "d", "e"];
        /// let func =
        ///     <Lcp2MmphfStr>::try_par_new(&keys, no_logging![])?.try_into_unaligned()?;
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

        /// Creates a two-step LCP-based monotone minimal perfect hash function for
        /// byte-sequence keys from a slice, using parallel hash computation
        /// and the given [`VBuilder`] configuration.
        ///
        /// This is the parallel counterpart of
        /// [`try_new_with_builder`].
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
        /// # use sux::func::{Lcp2MmphfStr, VBuilder};
        /// # use sux::traits::TryIntoUnaligned;
        /// # use dsi_progress_logger::no_logging;
        /// let keys = vec!["a", "b", "c", "d", "e"];
        /// let func = <Lcp2MmphfStr>::try_par_new_with_builder(
        ///     &keys,
        ///     VBuilder::default(),
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
        pub fn try_par_new_with_builder<
            B: AsRef<[u8]> + Borrow<K> + Sync,
            P: ProgressLog + Clone + Send + Sync,
        >(
            keys: &[B],
            builder: VBuilder<BitFieldVec<Box<[usize]>>, S0, E0>,
            pl: &mut P,
        ) -> Result<Self>
        where
            K: Sync,
        {
            let n = keys.len();
            if n == 0 {
                return Ok(Self::empty());
            }

            let (log2_bs, num_buckets) = bucket_layout(n, pl);

            // -- Sequential pass: compute bit-level LCPs --

            let mut scan = BytesLcpScan::<LcpLen>::new(log2_bs, num_buckets);
            for key in keys {
                scan.push(key.as_ref())?;
            }
            scan.finish()?;

            let lcp_bit_lens = &scan.lcp_bit_lens;
            let bucket_mask = (1 << log2_bs) - 1;
            builder.expected_num_keys(n).try_par_populate_and_build(
                keys,
                &|i| ((lcp_bit_lens[i >> log2_bs] as u64) << log2_bs) | (i & bucket_mask) as u64,
                &mut |builder,
                      seed,
                      mut store,
                      _max_value,
                      _num_keys,
                      pl: &mut P,
                      _state: &mut ()| {
                    Self::build_from_store(builder, seed, &mut *store, &scan, log2_bs, pl)
                },
                pl,
                (),
            )
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// `LcpLen` must hold LCP bit-lengths above 65535 on every target (it
        /// was previously u16 on 32-bit, silently wrapping long common
        /// prefixes).
        #[test]
        fn lcp_len_holds_large_prefixes() {
            assert!(
                u64::from(LcpLen::MAX) >= 65536,
                "LcpLen must hold LCP lengths above 65535 bits"
            );
        }

        /// The counts of LCP bit lengths take into account the size of the
        /// last, possibly partial, bucket.
        #[test]
        fn lcp_counts_partial_bucket() {
            let (counts, max_lcp) = lcp_counts(&[3, 5, 3], 2, 10);
            assert_eq!(counts.get(3), 6);
            assert_eq!(counts.get(5), 4);
            assert_eq!(max_lcp, 5);
        }
    }
} // mod build

/// A two-step monotone minimal perfect hash function for sorted
/// byte-sequence keys based on longest common prefixes (LCPs).
///
/// See the [module documentation] for the algorithmic description.
/// See [`Lcp2MmphfStr`] and [`Lcp2MmphfSliceU8`] for common instantiations,
/// and [`Lcp2MmphfInt`] for integer keys.
///
/// Instances of this structure are immutable; they are built using [`try_new`]
/// or one of its variants, and can be serialized using [ε-serde] or [`serde`].
/// The construction can be configured by passing a [`VBuilder`] to the
/// `*_with_builder` constructors. Since signatures are computed from the
/// native representation of data (see [`ToSig`]), a serialized function must
/// be queried on a machine with the same endianness.
///
/// This structure implements the [`TryIntoUnaligned`] trait, allowing it to be
/// converted into (usually faster) structures using unaligned access.
///
/// The `lcp_mmphf` binary (`cli` feature) builds and serializes instances of
/// [`Lcp2MmphfStr`] when invoked with the `--two-step` option.
///
/// # Alternatives
///
/// - [`LcpMmphf`] has slightly faster queries, but uses more space.
///
/// - [`PHastR`] is faster and uses less space if any bijection between the
///   keys and [0 . . *n*) is sufficient.
///
/// - A [`SignedFunc`] wrapping this structure rejects (with high
///   probability) keys outside the original set.
///
/// # Type parameters
///
/// - `K`: the byte-sequence key type; it must implement [`ToSig`] and
///   `AsRef<[u8]>` (e.g., `str` or `[u8]`).
/// - `D`: the backing store for [`VFunc`] data (e.g.,
///   [`BitFieldVec`]).
/// - `S0`: the [signature type] for the key maps (`lcp_freq_len_offset`
///   and `lcp_infreq_len`).
/// - `E0`: the [`ShardEdge`] for the key map `lcp_freq_len_offset`.
/// - `F0`: the [`ShardEdge`] for `lcp_infreq_len`. Defaults to
///   [`Fuse3Shards`].
/// - `S1`: the [signature type] for the prefix-to-bucket map
///   (`lcp_to_bucket`).
/// - `E1`: the [`ShardEdge`] for the prefix-to-bucket map.
///
/// # Examples
///
/// See [`try_new`]. See also [`Lcp2MmphfStr`] and [`Lcp2MmphfSliceU8`]
/// for common instantiations.
///
/// [module documentation]: self
/// [signature type]: Sig
/// [`try_new`]: Lcp2Mmphf::try_new
/// [ε-serde]: https://crates.io/crates/epserde
/// [`serde`]: https://crates.io/crates/serde
/// [`VBuilder`]: crate::func::VBuilder
/// [`LcpMmphf`]: crate::func::LcpMmphf
/// [`PHastR`]: crate::func::PHastR
/// [`SignedFunc`]: crate::func::SignedFunc
#[doc(alias("monotone", "mmphf"))]
#[derive(MemSize, MemDbg)]
#[cfg_attr(
    feature = "epserde",
    derive(epserde::Epserde),
    epserde(phantom(K, S0, S1))
)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "VFunc<K, D, S0, E0>: serde::Serialize, VFunc<K, D, S0, F0>: serde::Serialize, VFunc<BitPrefix, D, S1, E1>: serde::Serialize",
        deserialize = "VFunc<K, D, S0, E0>: serde::Deserialize<'de>, VFunc<K, D, S0, F0>: serde::Deserialize<'de>, VFunc<BitPrefix, D, S1, E1>: serde::Deserialize<'de>"
    ))
)]
pub struct Lcp2Mmphf<
    K: ?Sized,
    D = BitFieldVec<Box<[usize]>>,
    S0 = [u64; 2],
    E0 = FuseAug3Shards,
    F0 = Fuse3Shards,
    S1 = [u64; 2],
    E1 = FuseAug3Shards,
> {
    /// The number of keys.
    pub(crate) n: usize,
    /// The base-2 logarithm of the bucket size.
    pub(crate) log2_bucket_size: usize,
    /// Maps key → lcp_freq_len | offset.
    pub(crate) lcp_freq_len_offset: VFunc<K, D, S0, E0>,
    /// Maps escaped keys to their full LCP bit length.
    pub(crate) lcp_infreq_len: VFunc<K, D, S0, F0>,
    /// Maps frequent LCP indices back to actual LCP bit lengths.
    pub(crate) remap: Box<[usize]>,
    /// Escape sentinel for the frequent LCP part (2^r − 1).
    pub(crate) escape: usize,
    /// Maps each LCP bit-prefix to its bucket index.
    pub(crate) lcp_to_bucket: VFunc<BitPrefix, D, S1, E1>,
}

// Clone is implemented by hand, as a derived implementation would require K
// to implement Clone, too (e.g., str is not Clone)
impl<K: ?Sized, D: Clone, S0, E0: Clone, F0: Clone, S1, E1: Clone> Clone
    for Lcp2Mmphf<K, D, S0, E0, F0, S1, E1>
{
    fn clone(&self) -> Self {
        Self {
            n: self.n,
            log2_bucket_size: self.log2_bucket_size,
            lcp_freq_len_offset: self.lcp_freq_len_offset.clone(),
            lcp_infreq_len: self.lcp_infreq_len.clone(),
            remap: self.remap.clone(),
            escape: self.escape,
            lcp_to_bucket: self.lcp_to_bucket.clone(),
        }
    }
}

impl<K: ?Sized, D: SliceByValue, S0, E0, F0, S1, E1> std::fmt::Debug
    for Lcp2Mmphf<K, D, S0, E0, F0, S1, E1>
where
    VFunc<K, D, S0, E0>: std::fmt::Debug,
    VFunc<K, D, S0, F0>: std::fmt::Debug,
    VFunc<BitPrefix, D, S1, E1>: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Lcp2Mmphf")
            .field("n", &self.n)
            .field("log2_bucket_size", &self.log2_bucket_size)
            .field("lcp_freq_len_offset", &self.lcp_freq_len_offset)
            .field("lcp_infreq_len", &self.lcp_infreq_len)
            .field("remap", &self.remap)
            .field("escape", &self.escape)
            .field("lcp_to_bucket", &self.lcp_to_bucket)
            .finish()
    }
}

/// A [`Lcp2Mmphf`] for `str` keys.
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
/// # use sux::func::Lcp2MmphfStr;
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
///     <Lcp2MmphfStr>::try_new(FromSlice::new(&keys), keys.len(), no_logging![])?.try_into_unaligned()?;
///
/// for (i, key) in keys.iter().enumerate() {
///     assert_eq!(func.get(key.as_str()), i);
/// }
/// # Ok(())
/// # }
/// # #[cfg(not(feature = "rayon"))]
/// # fn main() {}
/// ```
pub type Lcp2MmphfStr<
    D = BitFieldVec<Box<[usize]>>,
    S0 = [u64; 2],
    E0 = FuseAug3Shards,
    F0 = Fuse3Shards,
    S1 = [u64; 2],
    E1 = FuseAug3Shards,
> = Lcp2Mmphf<str, D, S0, E0, F0, S1, E1>;
/// A [`Lcp2Mmphf`] for `[u8]` keys.
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
/// # use sux::func::Lcp2MmphfSliceU8;
/// # use sux::traits::TryIntoUnaligned;
/// # use sux::utils::FromSlice;
/// let keys: Vec<Vec<u8>> = vec![
///     b"alpha".to_vec(),
///     b"beta".to_vec(),
///     b"delta".to_vec(),
///     b"gamma".to_vec(),
/// ];
///
/// let func = <Lcp2MmphfSliceU8>::try_new(
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
pub type Lcp2MmphfSliceU8<
    D = BitFieldVec<Box<[usize]>>,
    S0 = [u64; 2],
    E0 = FuseAug3Shards,
    F0 = Fuse3Shards,
    S1 = [u64; 2],
    E1 = FuseAug3Shards,
> = Lcp2Mmphf<[u8], D, S0, E0, F0, S1, E1>;

impl<
    K: ?Sized + AsRef<[u8]> + ToSig<S0>,
    D: SliceByValue<Value = usize>,
    S0: Sig,
    E0: ShardEdge<S0, 3>,
    F0: ShardEdge<S0, 3>,
    S1: Sig,
    E1: ShardEdge<S1, 3>,
> Lcp2Mmphf<K, D, S0, E0, F0, S1, E1>
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
        let sig = K::to_sig(key, self.lcp_freq_len_offset.seed);
        self.get_by_sig(key, sig)
    }

    #[inline]
    pub(crate) fn get_by_sig(&self, key: &K, sig: S0) -> usize {
        let packed = self.lcp_freq_len_offset.get_by_sig(sig);
        let offset = packed & ((1 << self.log2_bucket_size) - 1);
        let frequent_lcp = packed >> self.log2_bucket_size;
        let lcp_bit_len = if frequent_lcp != self.escape {
            self.remap[frequent_lcp]
        } else {
            self.lcp_infreq_len.get_by_sig(sig)
        };

        let key_bytes: &[u8] = key.as_ref();
        let seed = self.lcp_to_bucket.seed;
        let sig: S1 = if lcp_bit_len <= key_bytes.len().saturating_mul(8) {
            bit_prefix_sig(key_bytes, lcp_bit_len, seed)
        } else {
            // Rare: LCP extends into the virtual NUL (at most 8 extra bits).
            // Since the NUL byte is 0x00, masking is a no-op, so we can
            // just hash all key bytes + the NUL + the bit length.
            let mut hasher = xxh3::Xxh3::with_seed(seed);
            hasher.update(key_bytes);
            hasher.update(&[0u8]);
            hasher.update(&lcp_bit_len.to_ne_bytes());
            S1::from_hasher(&hasher)
        };
        let bucket = self.lcp_to_bucket.get_by_sig(sig);
        (bucket << self.log2_bucket_size) + offset
    }
}

impl<K: ?Sized, D: SliceByValue, S0, E0, F0, S1, E1> Lcp2Mmphf<K, D, S0, E0, F0, S1, E1> {
    /// Returns the number of keys.
    pub const fn len(&self) -> usize {
        self.n
    }
    /// Returns `true` if the function contains no keys.
    pub const fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Returns the seed used to compute the signatures of the keys.
    pub const fn seed(&self) -> u64 {
        self.lcp_freq_len_offset.seed
    }
}

// ── Aligned ↔ Unaligned conversions ──────────────────────────────────

use crate::traits::{TryIntoUnaligned, Unaligned};

// -- Lcp2MmphfInt --

impl<K, S0: Sig, E0: ShardEdge<S0, 3>, F0: ShardEdge<S0, 3>, S1: Sig, E1: ShardEdge<S1, 3>>
    From<Unaligned<Lcp2MmphfInt<K, BitFieldVec<Box<[usize]>>, S0, E0, F0, S1, E1>>>
    for Lcp2MmphfInt<K, BitFieldVec<Box<[usize]>>, S0, E0, F0, S1, E1>
{
    fn from(f: Unaligned<Lcp2MmphfInt<K, BitFieldVec<Box<[usize]>>, S0, E0, F0, S1, E1>>) -> Self {
        Lcp2MmphfInt {
            n: f.n,
            log2_bucket_size: f.log2_bucket_size,
            lcp_freq_len_offset: f.lcp_freq_len_offset.into(),
            lcp_infreq_len: f.lcp_infreq_len.into(),
            remap: f.remap,
            escape: f.escape,
            lcp_to_bucket: f.lcp_to_bucket.into(),
        }
    }
}

impl<K, S0: Sig, E0: ShardEdge<S0, 3>, F0: ShardEdge<S0, 3>, S1: Sig, E1: ShardEdge<S1, 3>>
    TryIntoUnaligned for Lcp2MmphfInt<K, BitFieldVec<Box<[usize]>>, S0, E0, F0, S1, E1>
{
    type Unaligned = Lcp2MmphfInt<K, Unaligned<BitFieldVec<Box<[usize]>>>, S0, E0, F0, S1, E1>;
    fn try_into_unaligned(
        self,
    ) -> Result<Self::Unaligned, crate::traits::UnalignedConversionError> {
        Ok(Lcp2MmphfInt {
            n: self.n,
            log2_bucket_size: self.log2_bucket_size,
            lcp_freq_len_offset: self.lcp_freq_len_offset.try_into_unaligned()?,
            lcp_infreq_len: self.lcp_infreq_len.try_into_unaligned()?,
            remap: self.remap,
            escape: self.escape,
            lcp_to_bucket: self.lcp_to_bucket.try_into_unaligned()?,
        })
    }
}

// -- Lcp2Mmphf --

impl<K: ?Sized, S0: Sig, E0: ShardEdge<S0, 3>, F0: ShardEdge<S0, 3>, S1: Sig, E1: ShardEdge<S1, 3>>
    From<Unaligned<Lcp2Mmphf<K, BitFieldVec<Box<[usize]>>, S0, E0, F0, S1, E1>>>
    for Lcp2Mmphf<K, BitFieldVec<Box<[usize]>>, S0, E0, F0, S1, E1>
{
    fn from(f: Unaligned<Lcp2Mmphf<K, BitFieldVec<Box<[usize]>>, S0, E0, F0, S1, E1>>) -> Self {
        Lcp2Mmphf {
            n: f.n,
            log2_bucket_size: f.log2_bucket_size,
            lcp_freq_len_offset: f.lcp_freq_len_offset.into(),
            lcp_infreq_len: f.lcp_infreq_len.into(),
            remap: f.remap,
            escape: f.escape,
            lcp_to_bucket: f.lcp_to_bucket.into(),
        }
    }
}

impl<K: ?Sized, S0: Sig, E0: ShardEdge<S0, 3>, F0: ShardEdge<S0, 3>, S1: Sig, E1: ShardEdge<S1, 3>>
    TryIntoUnaligned for Lcp2Mmphf<K, BitFieldVec<Box<[usize]>>, S0, E0, F0, S1, E1>
{
    type Unaligned = Lcp2Mmphf<K, Unaligned<BitFieldVec<Box<[usize]>>>, S0, E0, F0, S1, E1>;
    fn try_into_unaligned(
        self,
    ) -> Result<Self::Unaligned, crate::traits::UnalignedConversionError> {
        Ok(Lcp2Mmphf {
            n: self.n,
            log2_bucket_size: self.log2_bucket_size,
            lcp_freq_len_offset: self.lcp_freq_len_offset.try_into_unaligned()?,
            lcp_infreq_len: self.lcp_infreq_len.try_into_unaligned()?,
            remap: self.remap,
            escape: self.escape,
            lcp_to_bucket: self.lcp_to_bucket.try_into_unaligned()?,
        })
    }
}
