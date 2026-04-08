/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

#![allow(clippy::type_complexity)]

//! Two-step LCP-based monotone minimal perfect hash functions.
//!
//! This module contains structures analogous to those in [`lcp_mmphf`];
//! however, they use a secondary [`VFunc`] for infrequent prefix lengths,
//! similarly to a [`VFunc2`], providing some space savings at the cost
//! of slightly slower queries.
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
//! # References
//!
//! Djamal Belazzougui, Paolo Boldi, Rasmus Pagh, and Sebastiano Vigna. [Theory
//! and practice of monotone minimal perfect
//! hashing](https://doi.org/10.1145/1963190.2025378). *ACM Journal of
//! Experimental Algorithmics*, 16(3):3.2:1−3.2:26, 2011.

use crate::bits::BitFieldVec;
use crate::func::VFunc;
use crate::func::lcp_mmphf::{BitPrefix, IntBitPrefix, bit_prefix_sig};
use crate::func::shard_edge::{Fuse3NoShards, FuseLge3Shards, ShardEdge};
#[cfg(feature = "rayon")]
use crate::func::vfunc2::{HybridMap, find_optimal_r};
use crate::utils::*;
use mem_dbg::*;
use num_primitive::PrimitiveInteger;
use value_traits::slices::SliceByValue;
use xxhash_rust::xxh3;

#[cfg(feature = "rayon")]
use {
    crate::func::VBuilder,
    crate::func::lcp_mmphf::{lcp_bits, lcp_bits_nul, log2_bucket_size},
    anyhow::{Result, bail},
    dsi_progress_logger::ProgressLog,
    lender::*,
    rdst::RadixKey,
    std::borrow::Borrow,
    sync_cell_slice::SyncSlice,
};

/// Compact type for LCP bit lengths. Using a smaller type than
/// `usize` reduces the cache footprint of the LCP lookup table,
/// which is on the hot path during peeling. `u32` on 64-bit
/// platforms supports keys up to 512 MB; `u16` on 32-bit supports
/// keys up to 8 KB.
#[cfg(all(feature = "rayon", target_pointer_width = "64"))]
type LcpLen = u32;
#[cfg(all(feature = "rayon", not(target_pointer_width = "64")))]
type LcpLen = u16;

/// A two-step monotone minimal perfect hash function for sorted integer keys based
/// on longest common bit-prefixes (LCPs).
///
/// See the [module documentation](self) for the algorithmic description.
///
/// This structure implements the [`TryIntoUnaligned`] trait, allowing it to be
/// converted into (usually faster) structures using unaligned access.
///
/// # Type parameters
///
/// - `K`: the integer key type.
/// - `D`: the backing store for [`VFunc`] data (e.g.,
///   [`BitFieldVec`]).
/// - `S0`: the [signature type](`Sig`) for the key maps (`fused` and
///   `lcp_long`).
/// - `E0`: the [`ShardEdge`] for the key maps (`fused`).
/// - `F0`: the [`ShardEdge`] for the long map (`lcp_long`). Defaults to
///   `E0`.
/// - `S1`: the  [signature type](`Sig`) for the prefix-to-bucket map
///   (`lcp2bucket`).
/// - `E1`: the [`ShardEdge`] for the prefix-to-bucket map.
///
/// # Examples
///
/// ```rust
/// # #[cfg(feature = "rayon")]
/// # fn main() -> anyhow::Result<()> {
/// # use dsi_progress_logger::no_logging;
/// # use sux::func::Lcp2MmphfInt;
/// # use sux::traits::TryIntoUnaligned;
/// # use sux::utils::FromSlice;
/// let keys: Vec<u64> = vec![10, 20, 30, 40, 50];
///
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
#[derive(MemDbg, MemSize)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde))]
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
    E0 = FuseLge3Shards,
    F0 = E0,
    S1 = [u64; 1],
    E1 = Fuse3NoShards,
> {
    /// The number of keys.
    pub(crate) n: usize,
    /// The base-2 logarithm of the bucket size.
    pub(crate) log2_bucket_size: usize,
    /// Fused function: maps key → (remapped_lcp << log2_bs) | offset.
    pub(crate) fused: VFunc<K, D, S0, E0>,
    /// Maps escaped keys to their full LCP bit length.
    pub(crate) lcp_long: VFunc<K, D, S0, F0>,
    /// Maps remapped LCP indices back to actual LCP bit lengths.
    pub(crate) remap: Box<[usize]>,
    /// Escape sentinel for the remapped LCP part (2^r − 1).
    pub(crate) escape: usize,
    /// Maps each LCP bit-prefix to its bucket index.
    pub(crate) lcp2bucket: VFunc<IntBitPrefix<K>, D, S1, E1>,
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
            .field("fused", &self.fused)
            .field("lcp_long", &self.lcp_long)
            .field("remap", &self.remap)
            .field("escape", &self.escape)
            .field("lcp2bucket", &self.lcp2bucket)
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
        let sig = K::to_sig(key, self.fused.seed);
        let packed = self.fused.get_by_sig(sig);
        let offset = packed & ((1 << self.log2_bucket_size) - 1);
        let remapped_lcp = packed >> self.log2_bucket_size;
        let lcp_bit_length = if remapped_lcp != self.escape {
            self.remap[remapped_lcp]
        } else {
            self.lcp_long.get_by_sig(sig)
        };
        let prefix = IntBitPrefix::new(key ^ K::MIN, lcp_bit_length);
        let bucket = self.lcp2bucket.get(prefix);
        (bucket << self.log2_bucket_size) + offset
    }
}

impl<
    K: PrimitiveInteger,
    D: SliceByValue,
    S0: Sig,
    E0: ShardEdge<S0, 3>,
    F0: ShardEdge<S0, 3>,
    S1: Sig,
    E1: ShardEdge<S1, 3>,
> Lcp2MmphfInt<K, D, S0, E0, F0, S1, E1>
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
    SigVal<E0::LocalSig, usize>: std::ops::BitXor + std::ops::BitXorAssign,
    SigVal<E0::LocalSig, u64>: std::ops::BitXor + std::ops::BitXorAssign,
    SigVal<F0::LocalSig, usize>: std::ops::BitXor + std::ops::BitXorAssign,
    SigVal<F0::LocalSig, u64>: std::ops::BitXor + std::ops::BitXorAssign,
    SigVal<S1, usize>: RadixKey,
    SigVal<E1::LocalSig, usize>: std::ops::BitXor + std::ops::BitXorAssign,
{
    /// Creates a two-step LCP-based MMPHF for integers using default
    /// [`VBuilder`] settings.
    ///
    /// This is a convenience wrapper around
    /// [`try_new_with_builder`](Self::try_new_with_builder). Use that
    /// method if you need to configure construction parameters such
    /// as offline mode, thread count, or sharding overhead.
    ///
    /// If keys are available as a slice, [`try_par_new`](Self::try_par_new)
    /// parallelizes the hash computation for faster construction.
    ///
    /// The keys must be provided in strictly increasing order.
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
    /// mode](VBuilder::offline), [thread count](VBuilder::max_num_threads),
    /// [sharding overhead](VBuilder::eps), and [PRNG seed](VBuilder::seed).
    ///
    /// See also [`try_par_new_with_builder`](Self::try_par_new_with_builder)
    /// for parallel hash computation from slices.
    ///
    /// The keys must be provided in strictly increasing order.
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
            return Ok((
                Self {
                    n: 0,
                    log2_bucket_size: 0,
                    fused: VFunc::empty(),
                    lcp_long: VFunc::empty(),
                    remap: Box::new([]),
                    escape: 0,
                    lcp2bucket: VFunc::empty(),
                },
                keys,
            ));
        }

        let log2_bs = log2_bucket_size(n);
        let bucket_size = 1usize << log2_bs;
        let bucket_mask = bucket_size - 1;
        let num_buckets = n.div_ceil(bucket_size);

        pl.info(format_args!(
            "Bucket size: 2^{log2_bs} = {bucket_size} ({num_buckets} buckets for {n} keys)"
        ));

        // State threaded through populate and build closures.
        struct State<K> {
            bucket_first_keys: Vec<K>,
            lcp_bit_lengths: Vec<LcpLen>,
            lcp_counts: HybridMap<usize, usize>,
            max_lcp: usize,
        }

        let mut builder = builder.expected_num_keys(n);
        let mut state = State::<K> {
            bucket_first_keys: Vec::with_capacity(num_buckets),
            lcp_bit_lengths: Vec::with_capacity(num_buckets),
            lcp_counts: HybridMap::new(None, 0),
            max_lcp: 0,
        };

        let mut rs = builder.retry_state(pl);

        loop {
            let seed = rs.next_seed();

            let result = {
                // Buffer for the signatures in the current bucket
                let mut buf: Vec<S0> = Vec::with_capacity(bucket_size);
                let mut prev_key: Option<K> = None;
                let mut curr_lcp_bits: usize = 0;
                let mut max_value: u64 = 0;
                let mut idx: usize = 0;

                state.bucket_first_keys.clear();
                state.lcp_bit_lengths.clear();
                state.lcp_counts = HybridMap::new(None, 0);
                state.max_lcp = 0;

                let mut populate = |seed: u64,
                                    push: &mut dyn FnMut(SigVal<S0, u64>) -> anyhow::Result<()>,
                                    pl: &mut P,
                                    state: &mut State<K>| {
                    while let Some(key) = keys.next()? {
                        pl.light_update();
                        let key: K = *key;

                        if let Some(prev) = prev_key {
                            if key <= prev {
                                bail!(
                                    "Keys are not in strictly increasing order at \
                                     position {idx}: {prev:?} >= {key:?}"
                                );
                            }
                        }

                        let offset = idx & bucket_mask;

                        // Start of a new bucket — flush the previous one.
                        if offset == 0 && idx > 0 {
                            let lcp = curr_lcp_bits;
                            state.lcp_bit_lengths.push(lcp as LcpLen);
                            state.max_lcp = state.max_lcp.max(lcp);
                            let bsize = buf.len();
                            state.lcp_counts.add(lcp, bsize);
                            for (i, &sig) in buf.iter().enumerate() {
                                let packed = ((lcp as u64) << log2_bs) | i as u64;
                                max_value = max_value.max(packed);
                                push(SigVal { sig, val: packed })?;
                            }
                            buf.clear();
                            curr_lcp_bits = K::BITS as usize;
                        } else if offset == 0 {
                            curr_lcp_bits = K::BITS as usize;
                        } else {
                            curr_lcp_bits = curr_lcp_bits.min(lcp_bits(key, prev_key.unwrap()));
                        }

                        if offset == 0 {
                            state.bucket_first_keys.push(key);
                        }

                        let sig = K::to_sig(key, seed);
                        buf.push(sig);
                        prev_key = Some(key);
                        idx += 1;
                    }

                    // Flush the last (possibly partial) bucket.
                    if !buf.is_empty() {
                        let lcp = curr_lcp_bits;
                        state.lcp_bit_lengths.push(lcp as LcpLen);
                        state.max_lcp = state.max_lcp.max(lcp);
                        let bsize = buf.len();
                        state.lcp_counts.add(lcp, bsize);
                        for (i, &sig) in buf.iter().enumerate() {
                            let packed = ((lcp as u64) << log2_bs) | i as u64;
                            max_value = max_value.max(packed);
                            push(SigVal { sig, val: packed })?;
                        }
                        buf.clear();
                    }

                    assert_eq!(idx, n, "Expected {n} keys but got {idx}");
                    assert_eq!(state.lcp_bit_lengths.len(), num_buckets);

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
                          state: &mut State<K>| {
                        let shard_edge = builder.shard_edge;
                        let store = &mut *store;

                        // -- Compute optimal r and remap/inv_map from LCP frequencies --
                        let counts =
                            std::mem::replace(&mut state.lcp_counts, HybridMap::new(None, 0));
                        let sorted_vals = counts.keys_by_desc_value();
                        let m = sorted_vals.len();
                        let n_keys = store.len();

                        let best_r = find_optimal_r(
                            n_keys,
                            state.max_lcp,
                            &sorted_vals,
                            |v| counts.get(v),
                            usize::BITS as usize,
                        );

                        let escape_usize = (1usize << best_r).wrapping_sub(1);
                        let num_remapped = escape_usize.min(m);

                        let remap: Box<[usize]> =
                            sorted_vals[..num_remapped].to_vec().into_boxed_slice();
                        let mut inv_map: HybridMap<usize, usize> =
                            HybridMap::new(Some(state.max_lcp), escape_usize);
                        for (i, &val) in remap.iter().enumerate() {
                            inv_map.insert(val, i);
                        }

                        let n_escaped = n_keys
                            - sorted_vals[..num_remapped]
                                .iter()
                                .map(|&v| counts.get(v))
                                .sum::<usize>();

                        pl.info(format_args!(
                            "Fused offset+LCP: r={best_r}, log2_bs={log2_bs}, \
                             escape={escape_usize}, {num_remapped} remapped, \
                             {m} distinct LCP values, {n_escaped} escaped keys ({:.1}%)",
                            100.0 * n_escaped as f64 / n_keys as f64
                        ));

                        // -- Build fused VFunc: (remapped_lcp << log2_bs) | offset --
                        let fused_max = (escape_usize << log2_bs) | bucket_mask;

                        let shb = shard_edge.shard_high_bits();
                        let num_shards_se = 1usize << shb;
                        let shard_mask = (1u64 << shb) - 1;
                        let mut escaped_counts = vec![0usize; num_shards_se];
                        let sync_counts = escaped_counts.as_sync_slice();

                        pl.info(format_args!(
                            "Building fused offset+LCP map ({} bits)...",
                            best_r + log2_bs
                        ));

                        let fused = builder.try_build_func_with_store_and_inspect::<K, u64>(
                            seed,
                            shard_edge,
                            fused_max,
                            store,
                            &|_, sig_val| {
                                let lcp = (sig_val.val >> log2_bs) as usize;
                                let offset = (sig_val.val as usize) & bucket_mask;
                                (inv_map.get(lcp) << log2_bs) | offset
                            },
                            &|sv: &SigVal<S0, u64>| {
                                let lcp = (sv.val >> log2_bs) as usize;
                                if inv_map.get(lcp) == escape_usize {
                                    let shard_idx = sv.sig.high_bits(shb, shard_mask) as usize;
                                    // SAFETY: each shard is processed by
                                    // exactly one thread.
                                    unsafe {
                                        let c = sync_counts[shard_idx].get();
                                        sync_counts[shard_idx].set(c + 1);
                                    }
                                }
                            },
                            pl,
                        )?;

                        // -- Build LCP long VFunc (escaped keys only) --
                        let lcp_long = if n_escaped > 0 {
                            let mut long_shard_edge = F0::default();
                            long_shard_edge.set_up_shards(n_escaped, builder.eps);
                            let long_shb = long_shard_edge.shard_high_bits();

                            let long_num_shards = 1usize << long_shb;
                            let filtered_shard_sizes = if long_num_shards >= num_shards_se {
                                escaped_counts
                            } else {
                                let per = num_shards_se / long_num_shards;
                                escaped_counts.chunks(per).map(|c| c.iter().sum()).collect()
                            };

                            pl.info(format_args!(
                                "Building LCP long map ({n_escaped} escaped \
                                 keys, {:.1}%)...",
                                100.0 * n_escaped as f64 / n_keys as f64
                            ));

                            let mut filtered_store = FilteredShardStore::new(
                                store,
                                long_shb,
                                |sv: &SigVal<S0, u64>| {
                                    inv_map.get((sv.val >> log2_bs) as usize) == escape_usize
                                },
                                filtered_shard_sizes,
                            );

                            VBuilder::<BitFieldVec<Box<[usize]>>, S0, F0>::default()
                                .set_from(builder)
                                .try_build_func_with_store::<K, u64>(
                                    seed,
                                    long_shard_edge,
                                    state.max_lcp,
                                    &mut filtered_store,
                                    &|_e, sig_val| (sig_val.val >> log2_bs) as usize,
                                    pl,
                                )?
                        } else {
                            VFunc::empty()
                        };

                        // -- lcp2bucket --
                        pl.info(format_args!(
                            "Building LCP prefix → bucket map ({num_buckets} buckets)..."
                        ));
                        let lcp2bucket = <VFunc<
                            IntBitPrefix<K>,
                            BitFieldVec<Box<[usize]>>,
                            S1,
                            E1,
                        >>::try_new_with_builder(
                            FromCloneableIntoIterator::new((0..num_buckets).map(|b| {
                                IntBitPrefix::new(
                                    state.bucket_first_keys[b] ^ K::MIN,
                                    state.lcp_bit_lengths[b] as usize,
                                )
                            })),
                            FromCloneableIntoIterator::new(0..num_buckets),
                            num_buckets,
                            VBuilder::default(),
                            pl,
                        )?;

                        let result = Self {
                            n,
                            log2_bucket_size: log2_bs,
                            fused,
                            lcp_long,
                            remap,
                            escape: escape_usize,
                            lcp2bucket,
                        };
                        let total = result.mem_size(SizeFlags::default()) * 8;
                        pl.info(format_args!(
                            "Actual bit cost per key: {:.2} ({total} bits for {n} keys)",
                            total as f64 / n as f64
                        ));
                        Ok(result)
                    },
                    pl,
                    &mut state,
                )
            };

            if let Some(r) = rs.handle_solve_result(result, pl)? {
                return Ok((r, keys));
            }

            keys = keys.rewind()?;
        }
    }

    /// Creates a two-step LCP-based monotone minimal perfect hash function for
    /// integers from a slice, using parallel hash computation and default
    /// [`VBuilder`] settings.
    ///
    /// This is the parallel counterpart of [`try_new`](Self::try_new).
    /// It is a convenience wrapper around
    /// [`try_par_new_with_builder`](Self::try_par_new_with_builder)
    /// with `VBuilder::default()`.
    ///
    /// The keys must be provided in strictly increasing order.
    ///
    /// # Examples
    ///
    /// If keys are produced sequentially (e.g., from a file), use
    /// [`try_new`](Self::try_new) instead.
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
    /// [`try_new_with_builder`](Self::try_new_with_builder).
    ///
    /// The keys must be provided in strictly increasing order.
    ///
    /// # Examples
    ///
    /// If keys are produced sequentially (e.g., from a file), use
    /// [`try_new_with_builder`](Self::try_new_with_builder) instead.
    /// ```rust
    /// # #[cfg(feature = "rayon")]
    /// # fn main() -> anyhow::Result<()> {
    /// # use sux::func::{Lcp2MmphfInt, VBuilder};
    /// # use sux::traits::TryIntoUnaligned;
    /// # use dsi_progress_logger::no_logging;
    /// let keys: Vec<u64> = vec![10, 20, 30, 40, 50];
    /// let func = Lcp2MmphfInt::<u64>::try_par_new_with_builder(
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
    pub fn try_par_new_with_builder(
        keys: &[K],
        builder: VBuilder<BitFieldVec<Box<[usize]>>, S0, E0>,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> Result<Self> {
        Self::try_par_new_inner(keys, builder, pl)
    }

    /// Internal parallel constructor for integer keys.
    pub(crate) fn try_par_new_inner<P: ProgressLog + Clone + Send + Sync>(
        keys: &[K],
        builder: VBuilder<BitFieldVec<Box<[usize]>>, S0, E0>,
        pl: &mut P,
    ) -> Result<Self> {
        let n = keys.len();
        if n == 0 {
            return Ok(Self {
                n: 0,
                log2_bucket_size: 0,
                fused: VFunc::empty(),
                lcp_long: VFunc::empty(),
                remap: Box::new([]),
                escape: 0,
                lcp2bucket: VFunc::empty(),
            });
        }

        let log2_bs = log2_bucket_size(n);
        let bucket_size = 1usize << log2_bs;
        let bucket_mask = bucket_size - 1;
        let num_buckets = n.div_ceil(bucket_size);

        pl.info(format_args!(
            "Bucket size: 2^{log2_bs} = {bucket_size} ({num_buckets} buckets for {n} keys)"
        ));

        // -- Sequential pass: compute bit-level LCPs and frequencies --

        let mut lcp_bit_lengths: Vec<LcpLen> = Vec::with_capacity(num_buckets);
        let mut bucket_first_keys: Vec<K> = Vec::with_capacity(num_buckets);
        let mut lcp_counts: HybridMap<usize, usize> = HybridMap::new(None, 0);
        let mut max_lcp: usize = 0;

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
                    let lcp = curr_lcp_bits;
                    lcp_bit_lengths.push(lcp as LcpLen);
                    max_lcp = max_lcp.max(lcp);
                    let bsize = bucket_size; // previous bucket was full
                    lcp_counts.add(lcp, bsize);
                }
                bucket_first_keys.push(key);
                curr_lcp_bits = K::BITS as usize;
            } else {
                curr_lcp_bits = curr_lcp_bits.min(lcp_bits(key, prev_key.unwrap()));
            }

            prev_key = Some(key);
        }

        // Flush the last (possibly partial) bucket.
        {
            let lcp = curr_lcp_bits;
            lcp_bit_lengths.push(lcp as LcpLen);
            max_lcp = max_lcp.max(lcp);
            let bsize = n - (num_buckets - 1) * bucket_size;
            lcp_counts.add(lcp, bsize);
        }
        assert_eq!(lcp_bit_lengths.len(), num_buckets);

        // -- Compute optimal r and remap/inv_map from LCP frequencies --

        let sorted_vals = lcp_counts.keys_by_desc_value();
        let m = sorted_vals.len();

        let best_r = find_optimal_r(
            n,
            max_lcp,
            &sorted_vals,
            |v| lcp_counts.get(v),
            usize::BITS as usize,
        );

        let escape_usize = (1usize << best_r).wrapping_sub(1);
        let num_remapped = escape_usize.min(m);

        let remap: Box<[usize]> = sorted_vals[..num_remapped].to_vec().into_boxed_slice();
        let mut inv_map: HybridMap<usize, usize> = HybridMap::new(Some(max_lcp), escape_usize);
        for (i, &val) in remap.iter().enumerate() {
            inv_map.insert(val, i);
        }

        let n_escaped = n - sorted_vals[..num_remapped]
            .iter()
            .map(|&v| lcp_counts.get(v))
            .sum::<usize>();

        pl.info(format_args!(
            "Fused offset+LCP: r={best_r}, log2_bs={log2_bs}, \
             escape={escape_usize}, {num_remapped} remapped, \
             {m} distinct LCP values, {n_escaped} escaped keys ({:.1}%)",
            100.0 * n_escaped as f64 / n as f64
        ));

        // -- Parallel build: fused + lcp_long + lcp2bucket --

        pl.info(format_args!(
            "Building fused offset+LCP map (parallel, {} bits)...",
            best_r + log2_bs
        ));

        let fused_result = builder.expected_num_keys(n).try_par_populate_and_build(
            keys,
            &|i| {
                ((lcp_bit_lengths[i >> log2_bs] as u64) << log2_bs)
                    | (i as u64 & bucket_mask as u64)
            },
            &mut |builder, seed, mut store, _max_value, _num_keys, pl: &mut P, _state: &mut ()| {
                let shard_edge = builder.shard_edge;
                let store = &mut *store;

                // -- Build fused VFunc: (remapped_lcp << log2_bs) | offset --
                let fused_max = (escape_usize << log2_bs) | bucket_mask;

                let shb = shard_edge.shard_high_bits();
                let num_shards_se = 1usize << shb;
                let shard_mask = (1u64 << shb) - 1;
                let mut escaped_counts = vec![0usize; num_shards_se];
                let sync_counts = escaped_counts.as_sync_slice();

                let fused = builder.try_build_func_with_store_and_inspect::<K, u64>(
                    seed,
                    shard_edge,
                    fused_max,
                    store,
                    &|_, sig_val| {
                        let lcp = (sig_val.val >> log2_bs) as usize;
                        let offset = (sig_val.val as usize) & bucket_mask;
                        (inv_map.get(lcp) << log2_bs) | offset
                    },
                    &|sv: &SigVal<S0, u64>| {
                        let lcp = (sv.val >> log2_bs) as usize;
                        if inv_map.get(lcp) == escape_usize {
                            let shard_idx = sv.sig.high_bits(shb, shard_mask) as usize;
                            // SAFETY: each shard is processed by
                            // exactly one thread.
                            unsafe {
                                let c = sync_counts[shard_idx].get();
                                sync_counts[shard_idx].set(c + 1);
                            }
                        }
                    },
                    pl,
                )?;

                // -- Build LCP long VFunc (escaped keys only) --
                let lcp_long = if n_escaped > 0 {
                    let mut long_shard_edge = F0::default();
                    long_shard_edge.set_up_shards(n_escaped, builder.eps);
                    let long_shb = long_shard_edge.shard_high_bits();

                    let long_num_shards = 1usize << long_shb;
                    let filtered_shard_sizes = if long_num_shards >= num_shards_se {
                        escaped_counts
                    } else {
                        let per = num_shards_se / long_num_shards;
                        escaped_counts.chunks(per).map(|c| c.iter().sum()).collect()
                    };

                    pl.info(format_args!(
                        "Building LCP long map ({n_escaped} escaped \
                             keys, {:.1}%)...",
                        100.0 * n_escaped as f64 / n as f64
                    ));

                    let mut filtered_store = FilteredShardStore::new(
                        store,
                        long_shb,
                        |sv: &SigVal<S0, u64>| {
                            inv_map.get((sv.val >> log2_bs) as usize) == escape_usize
                        },
                        filtered_shard_sizes,
                    );

                    VBuilder::<BitFieldVec<Box<[usize]>>, S0, F0>::default()
                        .set_from(builder)
                        .try_build_func_with_store::<K, u64>(
                            seed,
                            long_shard_edge,
                            max_lcp,
                            &mut filtered_store,
                            &|_e, sig_val| (sig_val.val >> log2_bs) as usize,
                            pl,
                        )?
                } else {
                    VFunc::empty()
                };

                // -- lcp2bucket --
                pl.info(format_args!(
                    "Building LCP prefix → bucket map ({num_buckets} buckets)..."
                ));
                let lcp2bucket = <VFunc<
                    IntBitPrefix<K>,
                    BitFieldVec<Box<[usize]>>,
                    S1,
                    E1,
                >>::try_new_with_builder(
                    FromCloneableIntoIterator::new((0..num_buckets).map(|b| {
                        IntBitPrefix::new(
                            bucket_first_keys[b] ^ K::MIN,
                            lcp_bit_lengths[b] as usize,
                        )
                    })),
                    FromCloneableIntoIterator::new(0..num_buckets),
                    num_buckets,
                    VBuilder::default(),
                    pl,
                )?;

                let result = Self {
                    n,
                    log2_bucket_size: log2_bs,
                    fused,
                    lcp_long,
                    remap: remap.clone(),
                    escape: escape_usize,
                    lcp2bucket,
                };
                let total = result.mem_size(SizeFlags::default()) * 8;
                pl.info(format_args!(
                    "Actual bit cost per key: {:.2} ({total} bits for {n} keys)",
                    total as f64 / n as f64
                ));
                Ok(result)
            },
            pl,
            (),
        )?;

        Ok(fused_result)
    }
}

/// A two-step monotone minimal perfect hash function for sorted
/// byte-sequence keys based on longest common prefixes (LCPs).
///
/// See the [module documentation](self) for the algorithmic description.
/// See [`Lcp2MmphfStr`] and [`Lcp2MmphfSliceU8`] for common instantiations,
/// and [`Lcp2MmphfInt`] for integer keys.
///
/// This structure implements the [`TryIntoUnaligned`] trait, allowing it to be
/// converted into (usually faster) structures using unaligned access.
///
/// # Type parameters
///
/// - `K`: the integer key type.
/// - `D`: the backing store for [`VFunc`] data (e.g.,
///   [`BitFieldVec`]).
/// - `S0`: the [signature type](`Sig`) for the key maps (`fused` and
///   `lcp_long`).
/// - `E0`: the [`ShardEdge`] for the key maps (`fused`).
/// - `F0`: the [`ShardEdge`] for the long map (`lcp_long`). Defaults to
///   `E0`.
/// - `S1`: the  [signature type](`Sig`) for the prefix-to-bucket map
///   (`lcp2bucket`).
/// - `E1`: the [`ShardEdge`] for the prefix-to-bucket map.
///
#[derive(MemDbg, MemSize)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde))]
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
    E0 = FuseLge3Shards,
    F0 = E0,
    S1 = [u64; 1],
    E1 = Fuse3NoShards,
> {
    pub(crate) n: usize,
    pub(crate) log2_bucket_size: usize,
    pub(crate) fused: VFunc<K, D, S0, E0>,
    pub(crate) lcp_long: VFunc<K, D, S0, F0>,
    pub(crate) remap: Box<[usize]>,
    pub(crate) escape: usize,
    pub(crate) lcp2bucket: VFunc<BitPrefix, D, S1, E1>,
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
            .field("fused", &self.fused)
            .field("lcp_long", &self.lcp_long)
            .field("remap", &self.remap)
            .field("escape", &self.escape)
            .field("lcp2bucket", &self.lcp2bucket)
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
    E0 = FuseLge3Shards,
    F0 = E0,
    S1 = [u64; 1],
    E1 = Fuse3NoShards,
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
    E0 = FuseLge3Shards,
    F0 = E0,
    S1 = [u64; 1],
    E1 = Fuse3NoShards,
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
        let sig = K::to_sig(key, self.fused.seed);
        let packed = self.fused.get_by_sig(sig);
        let offset = packed & ((1 << self.log2_bucket_size) - 1);
        let remapped_lcp = packed >> self.log2_bucket_size;
        let lcp_bit_length = if remapped_lcp != self.escape {
            self.remap[remapped_lcp]
        } else {
            self.lcp_long.get_by_sig(sig)
        };

        let key_bytes: &[u8] = key.as_ref();
        let lcp2b_seed = self.lcp2bucket.seed;
        let lcp2b_sig: S1 = if lcp_bit_length <= key_bytes.len() * 8 {
            bit_prefix_sig(key_bytes, lcp_bit_length, lcp2b_seed)
        } else {
            // Rare: LCP extends into the virtual NUL (at most 8 extra bits).
            let mut hasher = xxh3::Xxh3::with_seed(lcp2b_seed);
            hasher.update(key_bytes);
            hasher.update(&[0u8]);
            hasher.update(&lcp_bit_length.to_ne_bytes());
            S1::from_hasher(&hasher)
        };
        let bucket = self.lcp2bucket.get_by_sig(lcp2b_sig);
        (bucket << self.log2_bucket_size) + offset
    }
}

impl<
    K: ?Sized,
    D: SliceByValue,
    S0: Sig,
    E0: ShardEdge<S0, 3>,
    F0: ShardEdge<S0, 3>,
    S1: Sig,
    E1: ShardEdge<S1, 3>,
> Lcp2Mmphf<K, D, S0, E0, F0, S1, E1>
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
    SigVal<E0::LocalSig, usize>: std::ops::BitXor + std::ops::BitXorAssign,
    SigVal<E0::LocalSig, u64>: std::ops::BitXor + std::ops::BitXorAssign,
    SigVal<F0::LocalSig, usize>: std::ops::BitXor + std::ops::BitXorAssign,
    SigVal<F0::LocalSig, u64>: std::ops::BitXor + std::ops::BitXorAssign,
    SigVal<S1, usize>: RadixKey,
    SigVal<E1::LocalSig, usize>: std::ops::BitXor + std::ops::BitXorAssign,
{
    /// Creates a two-step LCP-based monotone minimal perfect hash function for
    /// byte-sequence keys using default [`VBuilder`] settings.
    ///
    /// This is a convenience wrapper around
    /// [`try_new_with_builder`](Self::try_new_with_builder). Use that
    /// method if you need to configure construction parameters such
    /// as offline mode, thread count, or sharding overhead.
    ///
    /// If keys are available as a slice, [`try_par_new`](Self::try_par_new)
    /// parallelizes the hash computation for faster construction.
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
    /// mode](VBuilder::offline), [thread count](VBuilder::max_num_threads),
    /// [sharding overhead](VBuilder::eps), and [PRNG seed](VBuilder::seed).
    ///
    /// See also [`try_par_new_with_builder`](Self::try_par_new_with_builder)
    /// for parallel hash computation from slices.
    ///
    /// The keys must be in strictly increasing lexicographic order.
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
            return Ok((
                Self {
                    n: 0,
                    log2_bucket_size: 0,
                    fused: VFunc::empty(),
                    lcp_long: VFunc::empty(),
                    remap: Box::new([]),
                    escape: 0,
                    lcp2bucket: VFunc::empty(),
                },
                keys,
            ));
        }

        let log2_bs = log2_bucket_size(n);
        let bucket_size = 1usize << log2_bs;
        let bucket_mask = bucket_size - 1;
        let num_buckets = n.div_ceil(bucket_size);

        pl.info(format_args!(
            "Bucket size: 2^{log2_bs} = {bucket_size} ({num_buckets} buckets for {n} keys)"
        ));

        // State threaded through populate and build closures.
        struct State {
            bucket_first_keys: Vec<Vec<u8>>,
            lcp_bit_lengths: Vec<LcpLen>,
            lcp_counts: HybridMap<usize, usize>,
            max_lcp: usize,
        }

        let mut builder = builder.expected_num_keys(n);
        let mut state = State {
            bucket_first_keys: Vec::with_capacity(num_buckets),
            lcp_bit_lengths: Vec::with_capacity(num_buckets),
            lcp_counts: HybridMap::new(None, 0),
            max_lcp: 0,
        };

        let mut rs = builder.retry_state(pl);

        loop {
            let seed = rs.next_seed();

            let result = {
                // Buffer for the signatures in the current bucket
                let mut buf: Vec<S0> = Vec::with_capacity(bucket_size);
                let mut prev_key: Vec<u8> = Vec::new();
                let mut curr_lcp_bits: usize = 0;
                let mut max_value: u64 = 0;
                let mut idx: usize = 0;

                state.bucket_first_keys.clear();
                state.lcp_bit_lengths.clear();
                state.lcp_counts = HybridMap::new(None, 0);
                state.max_lcp = 0;

                let mut populate = |seed: u64,
                                    push: &mut dyn FnMut(SigVal<S0, u64>) -> anyhow::Result<()>,
                                    pl: &mut P,
                                    state: &mut State| {
                    while let Some(key) = keys.next()? {
                        pl.light_update();
                        let key_bytes: &[u8] = key.as_ref();

                        if idx > 0 && key_bytes <= prev_key.as_slice() {
                            bail!(
                                "Keys are not in strictly increasing lexicographic order \
                                 at position {idx}"
                            );
                        }

                        let offset = idx & bucket_mask;

                        // Start of a new bucket — flush the previous one.
                        if offset == 0 && idx > 0 {
                            let lcp = curr_lcp_bits;
                            state.lcp_bit_lengths.push(lcp as LcpLen);
                            state.max_lcp = state.max_lcp.max(lcp);
                            let bsize = buf.len();
                            state.lcp_counts.add(lcp, bsize);
                            for (i, &sig) in buf.iter().enumerate() {
                                let packed = ((lcp as u64) << log2_bs) | i as u64;
                                max_value = max_value.max(packed);
                                push(SigVal { sig, val: packed })?;
                            }
                            buf.clear();
                            curr_lcp_bits = (key_bytes.len() + 1) * 8;
                        } else if offset == 0 {
                            curr_lcp_bits = (key_bytes.len() + 1) * 8;
                        } else {
                            curr_lcp_bits =
                                curr_lcp_bits.min(lcp_bits_nul::<true>(key_bytes, &prev_key));
                        }

                        if offset == 0 {
                            state.bucket_first_keys.push(key_bytes.to_vec());
                        }

                        let sig = K::to_sig(key.borrow(), seed);
                        buf.push(sig);
                        prev_key.clear();
                        prev_key.extend_from_slice(key_bytes);
                        idx += 1;
                    }

                    // Flush the last (possibly partial) bucket.
                    if !buf.is_empty() {
                        let lcp = curr_lcp_bits;
                        state.lcp_bit_lengths.push(lcp as LcpLen);
                        state.max_lcp = state.max_lcp.max(lcp);
                        let bsize = buf.len();
                        state.lcp_counts.add(lcp, bsize);
                        for (i, &sig) in buf.iter().enumerate() {
                            let packed = ((lcp as u64) << log2_bs) | i as u64;
                            max_value = max_value.max(packed);
                            push(SigVal { sig, val: packed })?;
                        }
                        buf.clear();
                    }

                    assert_eq!(idx, n, "Expected {n} keys but got {idx}");
                    assert_eq!(state.lcp_bit_lengths.len(), num_buckets);

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
                          state: &mut State| {
                        let shard_edge = builder.shard_edge;
                        let store = &mut *store;

                        // -- Compute optimal r and remap/inv_map from LCP frequencies --
                        let counts =
                            std::mem::replace(&mut state.lcp_counts, HybridMap::new(None, 0));
                        let sorted_vals = counts.keys_by_desc_value();
                        let m = sorted_vals.len();
                        let n_keys = store.len();

                        let best_r = find_optimal_r(
                            n_keys,
                            state.max_lcp,
                            &sorted_vals,
                            |v| counts.get(v),
                            usize::BITS as usize,
                        );

                        let escape_usize = (1usize << best_r).wrapping_sub(1);
                        let num_remapped = escape_usize.min(m);

                        let remap: Box<[usize]> =
                            sorted_vals[..num_remapped].to_vec().into_boxed_slice();
                        let mut inv_map: HybridMap<usize, usize> =
                            HybridMap::new(Some(state.max_lcp), escape_usize);
                        for (i, &val) in remap.iter().enumerate() {
                            inv_map.insert(val, i);
                        }

                        let n_escaped = n_keys
                            - sorted_vals[..num_remapped]
                                .iter()
                                .map(|&v| counts.get(v))
                                .sum::<usize>();

                        pl.info(format_args!(
                            "Fused offset+LCP: r={best_r}, log2_bs={log2_bs}, \
                             escape={escape_usize}, {num_remapped} remapped, \
                             {m} distinct LCP values, {n_escaped} escaped keys ({:.1}%)",
                            100.0 * n_escaped as f64 / n_keys as f64
                        ));

                        // -- Build fused VFunc: (remapped_lcp << log2_bs) | offset --
                        let fused_max = (escape_usize << log2_bs) | bucket_mask;

                        let shb = shard_edge.shard_high_bits();
                        let num_shards_se = 1usize << shb;
                        let shard_mask = (1u64 << shb) - 1;
                        let mut escaped_counts = vec![0usize; num_shards_se];
                        let sync_counts = escaped_counts.as_sync_slice();

                        pl.info(format_args!(
                            "Building fused offset+LCP map ({} bits)...",
                            best_r + log2_bs
                        ));

                        let fused = builder.try_build_func_with_store_and_inspect::<K, u64>(
                            seed,
                            shard_edge,
                            fused_max,
                            store,
                            &|_, sig_val| {
                                let lcp = (sig_val.val >> log2_bs) as usize;
                                let offset = (sig_val.val as usize) & bucket_mask;
                                (inv_map.get(lcp) << log2_bs) | offset
                            },
                            &|sv: &SigVal<S0, u64>| {
                                let lcp = (sv.val >> log2_bs) as usize;
                                if inv_map.get(lcp) == escape_usize {
                                    let shard_idx = sv.sig.high_bits(shb, shard_mask) as usize;
                                    // SAFETY: each shard is processed by
                                    // exactly one thread.
                                    unsafe {
                                        let c = sync_counts[shard_idx].get();
                                        sync_counts[shard_idx].set(c + 1);
                                    }
                                }
                            },
                            pl,
                        )?;

                        // -- Build LCP long VFunc (escaped keys only) --
                        let lcp_long = if n_escaped > 0 {
                            let mut long_shard_edge = F0::default();
                            long_shard_edge.set_up_shards(n_escaped, builder.eps);
                            let long_shb = long_shard_edge.shard_high_bits();

                            let long_num_shards = 1usize << long_shb;
                            let filtered_shard_sizes = if long_num_shards >= num_shards_se {
                                escaped_counts
                            } else {
                                let per = num_shards_se / long_num_shards;
                                escaped_counts.chunks(per).map(|c| c.iter().sum()).collect()
                            };

                            pl.info(format_args!(
                                "Building LCP long map ({n_escaped} escaped \
                                 keys, {:.1}%)...",
                                100.0 * n_escaped as f64 / n_keys as f64
                            ));

                            let mut filtered_store = FilteredShardStore::new(
                                store,
                                long_shb,
                                |sv: &SigVal<S0, u64>| {
                                    inv_map.get((sv.val >> log2_bs) as usize) == escape_usize
                                },
                                filtered_shard_sizes,
                            );

                            VBuilder::<BitFieldVec<Box<[usize]>>, S0, F0>::default()
                                .set_from(builder)
                                .try_build_func_with_store::<K, u64>(
                                    seed,
                                    long_shard_edge,
                                    state.max_lcp,
                                    &mut filtered_store,
                                    &|_e, sig_val| (sig_val.val >> log2_bs) as usize,
                                    pl,
                                )?
                        } else {
                            VFunc::empty()
                        };

                        // -- lcp2bucket --
                        pl.info(format_args!(
                            "Building LCP prefix → bucket map ({num_buckets} buckets)..."
                        ));
                        let extended_first_keys: Vec<Vec<u8>> = state
                            .bucket_first_keys
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
                            S1,
                            E1,
                        >>::try_new_with_builder(
                            FromCloneableIntoIterator::new((0..num_buckets).map(|b| {
                                BitPrefix::new(
                                    &extended_first_keys[b],
                                    state.lcp_bit_lengths[b] as usize,
                                )
                            })),
                            FromCloneableIntoIterator::new(0..num_buckets),
                            num_buckets,
                            VBuilder::default(),
                            pl,
                        )?;

                        let result = Self {
                            n,
                            log2_bucket_size: log2_bs,
                            fused,
                            lcp_long,
                            remap,
                            escape: escape_usize,
                            lcp2bucket,
                        };
                        let total = result.mem_size(SizeFlags::default()) * 8;
                        pl.info(format_args!(
                            "Actual bit cost per key: {:.2} ({total} bits for {n} keys)",
                            total as f64 / n as f64
                        ));
                        Ok(result)
                    },
                    pl,
                    &mut state,
                )
            };

            if let Some(r) = rs.handle_solve_result(result, pl)? {
                return Ok((r, keys));
            }

            keys = keys.rewind()?;
        }
    }

    /// Creates a two-step LCP-based monotone minimal perfect hash function for
    /// byte-sequence keys from a slice, using parallel hash computation and
    /// default [`VBuilder`] settings.
    ///
    /// This is the parallel counterpart of [`try_new`](Self::try_new).
    /// It is a convenience wrapper around
    /// [`try_par_new_with_builder`](Self::try_par_new_with_builder)
    /// with `VBuilder::default()`.
    ///
    /// The keys must be in strictly increasing lexicographic order.
    ///
    /// # Examples
    ///
    /// If keys are produced sequentially (e.g., from a file), use
    /// [`try_new`](Self::try_new) instead.
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
    /// byte-sequence keys from a slice, using parallel hash computation and the
    /// given [`VBuilder`] configuration.
    ///
    /// This is the parallel counterpart of
    /// [`try_new_with_builder`](Self::try_new_with_builder).
    ///
    /// The keys must be in strictly increasing lexicographic order.
    ///
    /// # Examples
    ///
    /// If keys are produced sequentially (e.g., from a file), use
    /// [`try_new_with_builder`](Self::try_new_with_builder) instead.
    /// ```rust
    /// # #[cfg(feature = "rayon")]
    /// # fn main() -> anyhow::Result<()> {
    /// # use sux::func::{Lcp2MmphfStr, VBuilder};
    /// # use sux::traits::TryIntoUnaligned;
    /// # use dsi_progress_logger::no_logging;
    /// let keys = vec!["a", "b", "c", "d", "e"];
    /// let func = <Lcp2MmphfStr>::try_par_new_with_builder(
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
    pub(crate) fn try_par_new_inner<
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
            return Ok(Self {
                n: 0,
                log2_bucket_size: 0,
                fused: VFunc::empty(),
                lcp_long: VFunc::empty(),
                remap: Box::new([]),
                escape: 0,
                lcp2bucket: VFunc::empty(),
            });
        }

        let log2_bs = log2_bucket_size(n);
        let bucket_size = 1usize << log2_bs;
        let bucket_mask = bucket_size - 1;
        let num_buckets = n.div_ceil(bucket_size);

        pl.info(format_args!(
            "Bucket size: 2^{log2_bs} = {bucket_size} ({num_buckets} buckets for {n} keys)"
        ));

        // -- Sequential pass: compute bit-level LCPs and frequencies --

        let mut lcp_bit_lengths: Vec<LcpLen> = Vec::with_capacity(num_buckets);
        let mut bucket_first_keys: Vec<Vec<u8>> = Vec::with_capacity(num_buckets);
        let mut lcp_counts: HybridMap<usize, usize> = HybridMap::new(None, 0);
        let mut max_lcp: usize = 0;

        let mut prev_key: Vec<u8> = Vec::new();
        let mut curr_lcp_bits: usize = 0;

        for (i, key) in keys.iter().enumerate() {
            let key_bytes: &[u8] = key.as_ref();

            if i > 0 && key_bytes <= prev_key.as_slice() {
                bail!(
                    "Keys are not in strictly increasing lexicographic order \
                     at position {i}"
                );
            }

            let offset = i & bucket_mask;

            if offset == 0 {
                // First key of a new bucket.
                if i > 0 {
                    let lcp = curr_lcp_bits;
                    lcp_bit_lengths.push(lcp as LcpLen);
                    max_lcp = max_lcp.max(lcp);
                    let bsize = bucket_size; // previous bucket was full
                    lcp_counts.add(lcp, bsize);
                }
                bucket_first_keys.push(key_bytes.to_vec());
                curr_lcp_bits = (key_bytes.len() + 1) * 8;
            } else {
                curr_lcp_bits = curr_lcp_bits.min(lcp_bits_nul::<true>(key_bytes, &prev_key));
            }

            prev_key.clear();
            prev_key.extend_from_slice(key_bytes);
        }

        // Flush the last (possibly partial) bucket.
        {
            let lcp = curr_lcp_bits;
            lcp_bit_lengths.push(lcp as LcpLen);
            max_lcp = max_lcp.max(lcp);
            let bsize = n - (num_buckets - 1) * bucket_size;
            lcp_counts.add(lcp, bsize);
        }
        assert_eq!(lcp_bit_lengths.len(), num_buckets);

        // -- Compute optimal r and remap/inv_map from LCP frequencies --

        let sorted_vals = lcp_counts.keys_by_desc_value();
        let m = sorted_vals.len();

        let best_r = find_optimal_r(
            n,
            max_lcp,
            &sorted_vals,
            |v| lcp_counts.get(v),
            usize::BITS as usize,
        );

        let escape_usize = (1usize << best_r).wrapping_sub(1);
        let num_remapped = escape_usize.min(m);

        let remap: Box<[usize]> = sorted_vals[..num_remapped].to_vec().into_boxed_slice();
        let mut inv_map: HybridMap<usize, usize> = HybridMap::new(Some(max_lcp), escape_usize);
        for (i, &val) in remap.iter().enumerate() {
            inv_map.insert(val, i);
        }

        let n_escaped = n - sorted_vals[..num_remapped]
            .iter()
            .map(|&v| lcp_counts.get(v))
            .sum::<usize>();

        pl.info(format_args!(
            "Fused offset+LCP: r={best_r}, log2_bs={log2_bs}, \
             escape={escape_usize}, {num_remapped} remapped, \
             {m} distinct LCP values, {n_escaped} escaped keys ({:.1}%)",
            100.0 * n_escaped as f64 / n as f64
        ));

        // -- Parallel build: fused + lcp_long + lcp2bucket --

        pl.info(format_args!(
            "Building fused offset+LCP map (parallel, {} bits)...",
            best_r + log2_bs
        ));

        let fused_result = builder.expected_num_keys(n).try_par_populate_and_build(
            keys,
            &|i| {
                ((lcp_bit_lengths[i >> log2_bs] as u64) << log2_bs)
                    | (i as u64 & bucket_mask as u64)
            },
            &mut |builder, seed, mut store, _max_value, _num_keys, pl: &mut P, _state: &mut ()| {
                let shard_edge = builder.shard_edge;
                let store = &mut *store;

                // -- Build fused VFunc: (remapped_lcp << log2_bs) | offset --
                let fused_max = (escape_usize << log2_bs) | bucket_mask;

                let shb = shard_edge.shard_high_bits();
                let num_shards_se = 1usize << shb;
                let shard_mask = (1u64 << shb) - 1;
                let mut escaped_counts = vec![0usize; num_shards_se];
                let sync_counts = escaped_counts.as_sync_slice();

                let fused = builder.try_build_func_with_store_and_inspect::<K, u64>(
                    seed,
                    shard_edge,
                    fused_max,
                    store,
                    &|_, sig_val| {
                        let lcp = (sig_val.val >> log2_bs) as usize;
                        let offset = (sig_val.val as usize) & bucket_mask;
                        (inv_map.get(lcp) << log2_bs) | offset
                    },
                    &|sv: &SigVal<S0, u64>| {
                        let lcp = (sv.val >> log2_bs) as usize;
                        if inv_map.get(lcp) == escape_usize {
                            let shard_idx = sv.sig.high_bits(shb, shard_mask) as usize;
                            // SAFETY: each shard is processed by
                            // exactly one thread.
                            unsafe {
                                let c = sync_counts[shard_idx].get();
                                sync_counts[shard_idx].set(c + 1);
                            }
                        }
                    },
                    pl,
                )?;

                // -- Build LCP long VFunc (escaped keys only) --
                let lcp_long = if n_escaped > 0 {
                    let mut long_shard_edge = F0::default();
                    long_shard_edge.set_up_shards(n_escaped, builder.eps);
                    let long_shb = long_shard_edge.shard_high_bits();

                    let long_num_shards = 1usize << long_shb;
                    let filtered_shard_sizes = if long_num_shards >= num_shards_se {
                        escaped_counts
                    } else {
                        let per = num_shards_se / long_num_shards;
                        escaped_counts.chunks(per).map(|c| c.iter().sum()).collect()
                    };

                    pl.info(format_args!(
                        "Building LCP long map ({n_escaped} escaped \
                             keys, {:.1}%)...",
                        100.0 * n_escaped as f64 / n as f64
                    ));

                    let mut filtered_store = FilteredShardStore::new(
                        store,
                        long_shb,
                        |sv: &SigVal<S0, u64>| {
                            inv_map.get((sv.val >> log2_bs) as usize) == escape_usize
                        },
                        filtered_shard_sizes,
                    );

                    VBuilder::<BitFieldVec<Box<[usize]>>, S0, F0>::default()
                        .set_from(builder)
                        .try_build_func_with_store::<K, u64>(
                            seed,
                            long_shard_edge,
                            max_lcp,
                            &mut filtered_store,
                            &|_e, sig_val| (sig_val.val >> log2_bs) as usize,
                            pl,
                        )?
                } else {
                    VFunc::empty()
                };

                // -- lcp2bucket --
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

                let lcp2bucket =
                    <VFunc<BitPrefix, BitFieldVec<Box<[usize]>>, S1, E1>>::try_new_with_builder(
                        FromCloneableIntoIterator::new((0..num_buckets).map(|b| {
                            BitPrefix::new(&extended_first_keys[b], lcp_bit_lengths[b] as usize)
                        })),
                        FromCloneableIntoIterator::new(0..num_buckets),
                        num_buckets,
                        VBuilder::default(),
                        pl,
                    )?;

                let result = Self {
                    n,
                    log2_bucket_size: log2_bs,
                    fused,
                    lcp_long,
                    remap: remap.clone(),
                    escape: escape_usize,
                    lcp2bucket,
                };
                let total = result.mem_size(SizeFlags::default()) * 8;
                pl.info(format_args!(
                    "Actual bit cost per key: {:.2} ({total} bits for {n} keys)",
                    total as f64 / n as f64
                ));
                Ok(result)
            },
            pl,
            (),
        )?;

        Ok(fused_result)
    }
}

// ── Aligned ↔ Unaligned conversions ──────────────────────────────────

use crate::traits::{TryIntoUnaligned, Unaligned};
type Ubfv = Unaligned<BitFieldVec<Box<[usize]>>>;

// -- Lcp2MmphfInt --

impl<K, S0: Sig, E0: ShardEdge<S0, 3>, F0: ShardEdge<S0, 3>, S1: Sig, E1: ShardEdge<S1, 3>>
    From<Lcp2MmphfInt<K, Ubfv, S0, E0, F0, S1, E1>>
    for Lcp2MmphfInt<K, BitFieldVec<Box<[usize]>>, S0, E0, F0, S1, E1>
{
    fn from(f: Lcp2MmphfInt<K, Ubfv, S0, E0, F0, S1, E1>) -> Self {
        Lcp2MmphfInt {
            n: f.n,
            log2_bucket_size: f.log2_bucket_size,
            fused: f.fused.into(),
            lcp_long: f.lcp_long.into(),
            remap: f.remap,
            escape: f.escape,
            lcp2bucket: f.lcp2bucket.into(),
        }
    }
}

impl<K, S0: Sig, E0: ShardEdge<S0, 3>, F0: ShardEdge<S0, 3>, S1: Sig, E1: ShardEdge<S1, 3>>
    TryIntoUnaligned for Lcp2MmphfInt<K, BitFieldVec<Box<[usize]>>, S0, E0, F0, S1, E1>
{
    type Unaligned = Lcp2MmphfInt<K, Ubfv, S0, E0, F0, S1, E1>;
    fn try_into_unaligned(
        self,
    ) -> Result<Self::Unaligned, crate::traits::UnalignedConversionError> {
        Ok(Lcp2MmphfInt {
            n: self.n,
            log2_bucket_size: self.log2_bucket_size,
            fused: self.fused.try_into_unaligned()?,
            lcp_long: self.lcp_long.try_into_unaligned()?,
            remap: self.remap,
            escape: self.escape,
            lcp2bucket: self.lcp2bucket.try_into_unaligned()?,
        })
    }
}

// -- Lcp2Mmphf --

impl<K: ?Sized, S0: Sig, E0: ShardEdge<S0, 3>, F0: ShardEdge<S0, 3>, S1: Sig, E1: ShardEdge<S1, 3>>
    From<Lcp2Mmphf<K, Ubfv, S0, E0, F0, S1, E1>>
    for Lcp2Mmphf<K, BitFieldVec<Box<[usize]>>, S0, E0, F0, S1, E1>
{
    fn from(f: Lcp2Mmphf<K, Ubfv, S0, E0, F0, S1, E1>) -> Self {
        Lcp2Mmphf {
            n: f.n,
            log2_bucket_size: f.log2_bucket_size,
            fused: f.fused.into(),
            lcp_long: f.lcp_long.into(),
            remap: f.remap,
            escape: f.escape,
            lcp2bucket: f.lcp2bucket.into(),
        }
    }
}

impl<K: ?Sized, S0: Sig, E0: ShardEdge<S0, 3>, F0: ShardEdge<S0, 3>, S1: Sig, E1: ShardEdge<S1, 3>>
    TryIntoUnaligned for Lcp2Mmphf<K, BitFieldVec<Box<[usize]>>, S0, E0, F0, S1, E1>
{
    type Unaligned = Lcp2Mmphf<K, Ubfv, S0, E0, F0, S1, E1>;
    fn try_into_unaligned(
        self,
    ) -> Result<Self::Unaligned, crate::traits::UnalignedConversionError> {
        Ok(Lcp2Mmphf {
            n: self.n,
            log2_bucket_size: self.log2_bucket_size,
            fused: self.fused.try_into_unaligned()?,
            lcp_long: self.lcp_long.try_into_unaligned()?,
            remap: self.remap,
            escape: self.escape,
            lcp2bucket: self.lcp2bucket.try_into_unaligned()?,
        })
    }
}
