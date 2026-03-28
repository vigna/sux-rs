/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

#![allow(clippy::type_complexity)]

//! Two-step LCP-based monotone minimal perfect hash functions.
//!
//! Compared to [`LcpMmphfInt`](super::LcpMmphfInt) /
//! [`LcpMmphf`](super::LcpMmphf), these variants use a [`VFunc2`] for the
//! LCP-length component, trading ≈3 extra independent random memory accesses
//! per query for ≈20–35% less space.
//!
//! See [`Lcp2MmphfInt`], [`Lcp2MmphfStr`], and [`Lcp2MmphfSliceU8`].
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
use crate::func::shard_edge::{Fuse3NoShards, Fuse3Shards, FuseLge3Shards, ShardEdge};
use crate::func::vfunc2::VFunc2;
use crate::utils::*;
use mem_dbg::*;
use num_primitive::PrimitiveInteger;
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
    std::cell::RefCell,
};

// ── Integer variant ─────────────────────────────────────────────────

/// A two-step monotone minimal perfect hash function for sorted integers.
///
/// Like [`LcpMmphfInt`](super::LcpMmphfInt) but uses a [`VFunc2`] for
/// the LCP-length component, trading speed for less space.
///
/// # Examples
///
/// ```rust
/// # #[cfg(feature = "rayon")]
/// # fn main() -> anyhow::Result<()> {
/// # use dsi_progress_logger::no_logging;
/// # use sux::func::Lcp2MmphfInt;
/// # use sux::utils::FromSlice;
/// let keys: Vec<u64> = vec![10, 20, 30, 40, 50];
///
/// let func: Lcp2MmphfInt<u64> =
///     Lcp2MmphfInt::try_new(FromSlice::new(&keys), keys.len(), no_logging![])?;
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
pub struct Lcp2MmphfInt<T: PrimitiveInteger, S = [u64; 2], E = FuseLge3Shards> {
    pub(crate) n: usize,
    pub(crate) log2_bucket_size: usize,
    /// Maps each key to its offset within the bucket.
    pub(crate) offsets: VFunc<T, usize, BitFieldVec<Box<[usize]>>, S, E>,
    /// Two-step retrieval of LCP lengths.
    pub(crate) lcp_lengths: VFunc2<T, usize, BitFieldVec<Box<[usize]>>, S, E, Fuse3Shards>,
    /// Maps each LCP bit-prefix to its bucket index.
    pub(crate) lcp2bucket:
        VFunc<IntBitPrefix<T>, usize, BitFieldVec<Box<[usize]>>, [u64; 1], Fuse3NoShards>,
}

impl<T: PrimitiveInteger + ToSig<S>, S: Sig, E: ShardEdge<S, 3>> Lcp2MmphfInt<T, S, E>
where
    Fuse3Shards: ShardEdge<S, 3>,
{
    #[inline]
    pub fn get(&self, key: T) -> usize
    where
        T: Copy,
    {
        let sig = T::to_sig(key, self.offsets.seed);
        let offset = self.offsets.get_by_sig_unaligned(sig);
        let lcp_bit_length = self.lcp_lengths.get_by_sig_unaligned(sig);
        let prefix = IntBitPrefix::new(key ^ T::MIN, lcp_bit_length);
        let bucket = self.lcp2bucket.get_unaligned(prefix);
        (bucket << self.log2_bucket_size) + offset
    }
}

impl<T: PrimitiveInteger, S: Sig, E: ShardEdge<S, 3>> Lcp2MmphfInt<T, S, E>
where
    Fuse3Shards: ShardEdge<S, 3>,
{
    pub const fn len(&self) -> usize {
        self.n
    }
    pub const fn is_empty(&self) -> bool {
        self.n == 0
    }
}

#[cfg(feature = "rayon")]
impl<T, S, E> Lcp2MmphfInt<T, S, E>
where
    T: PrimitiveInteger + ToSig<S> + std::fmt::Debug + Send + Sync + Copy + Ord,
    S: Sig + Send + Sync,
    E: ShardEdge<S, 3>,
    Fuse3Shards: ShardEdge<S, 3>,
    SigVal<S, usize>: RadixKey,
    SigVal<E::LocalSig, usize>: std::ops::BitXor + std::ops::BitXorAssign,
    SigVal<<Fuse3Shards as ShardEdge<S, 3>>::LocalSig, usize>:
        std::ops::BitXor + std::ops::BitXorAssign,
{
    /// Creates a two-step LCP-based MMPHF for integers using default
    /// [`VBuilder`] settings.
    ///
    /// This is a convenience wrapper around
    /// [`try_new_with_builder`](Self::try_new_with_builder). Use that
    /// method if you need to configure construction parameters such
    /// as offline mode, thread count, or sharding overhead.
    ///
    /// The keys must be provided in strictly increasing order.
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

    /// Creates a two-step LCP-based MMPHF for integers using the
    /// given [`VBuilder`] configuration.
    ///
    /// The builder controls construction parameters such as offline
    /// mode (`offline`), thread count (`max_num_threads`), sharding
    /// overhead (`eps`), and PRNG seed (`seed`).
    ///
    /// The keys must be provided in strictly increasing order.
    pub fn try_new_with_builder<P: ProgressLog + Clone + Send + Sync>(
        keys: impl FallibleRewindableLender<
            RewindError: std::error::Error + Send + Sync + 'static,
            Error: std::error::Error + Send + Sync + 'static,
        > + for<'lend> FallibleLending<'lend, Lend = &'lend T>,
        n: usize,
        builder: VBuilder<usize, BitFieldVec<Box<[usize]>>, S, E>,
        pl: &mut P,
    ) -> Result<Self> {
        if n == 0 {
            return Ok(Self {
                n: 0,
                log2_bucket_size: 0,
                offsets: VFunc::empty(),
                lcp_lengths: VFunc2::empty(),
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

        // LCP state shared between key_to_val and build_fn via RefCell.
        // (lcp_bit_lengths, bucket_first_keys, prev_key, curr_lcp_bits)
        let state = RefCell::new((
            Vec::<usize>::with_capacity(num_buckets),
            Vec::<T>::with_capacity(num_buckets),
            None::<T>,
            0usize,
        ));

        let mut builder = builder.expected_num_keys(n);
        builder.try_populate_and_build_with_fn(
            keys,
            &mut |key: &T, idx: usize| -> anyhow::Result<usize> {
                let mut s = state.borrow_mut();
                let (lcp_bit_lengths, bucket_first_keys, prev_key, curr_lcp_bits) = &mut *s;
                let key: T = *key;

                if idx == 0 {
                    lcp_bit_lengths.clear();
                    bucket_first_keys.clear();
                    *prev_key = None;
                    *curr_lcp_bits = 0;
                }

                if let Some(prev) = *prev_key {
                    if key <= prev {
                        bail!(
                            "Keys are not in strictly increasing order at position {idx}: \
                             {prev:?} >= {key:?}"
                        );
                    }
                }

                let offset = idx & bucket_mask;
                if offset == 0 {
                    if idx > 0 {
                        lcp_bit_lengths.push(*curr_lcp_bits);
                    }
                    bucket_first_keys.push(key);
                    *curr_lcp_bits = T::BITS as usize;
                } else {
                    *curr_lcp_bits = (*curr_lcp_bits).min(lcp_bits(key, prev_key.unwrap()));
                }
                *prev_key = Some(key);
                Ok(idx)
            },
            &mut |_builder, seed, store, _max_value, _num_keys, pl: &mut P| {
                let shard_edge = _builder.shard_edge;
                // Finalize LCP data (last bucket).
                {
                    let mut s = state.borrow_mut();
                    let last_lcp = s.3;
                    s.0.push(last_lcp);
                }
                let s = state.borrow();
                let (lcp_bit_lengths, bucket_first_keys, _, _) = &*s;
                assert_eq!(lcp_bit_lengths.len(), num_buckets);

                macro_rules! build_from_store {
                    ($store:expr) => {{
                        let store = $store;
                        // -- Offsets --
                        pl.info(format_args!(
                            "Building key → offset map ({log2_bs} bits)..."
                        ));
                        let offsets = VBuilder::<usize, BitFieldVec<Box<[usize]>>, S, E>::default()
                            .try_build_func_with_store::<T, usize>(
                                seed,
                                shard_edge,
                                bucket_mask,
                                store,
                                &|_, sig_val| sig_val.val & bucket_mask,
                                &|_| {},
                                pl,
                            )?;

                        // -- LCP lengths (two-step) --
                        // Compute frequency table directly from lcp_bit_lengths
                        // (no store scan needed).
                        let max_lcp = *lcp_bit_lengths.iter().max().unwrap_or(&0);
                        let mut lcp_counts: std::collections::HashMap<usize, usize> =
                            std::collections::HashMap::new();
                        let last_bucket_size = n - (num_buckets - 1) * bucket_size;
                        for (b, &lcp) in lcp_bit_lengths.iter().enumerate() {
                            let bsize = if b == num_buckets - 1 {
                                last_bucket_size
                            } else {
                                bucket_size
                            };
                            *lcp_counts.entry(lcp).or_insert(0) += bsize;
                        }

                        pl.info(format_args!("Building two-step LCP lengths..."));
                        let lcp_lengths = VFunc2::try_build_from_store_with_freq::<usize>(
                            seed,
                            shard_edge,
                            store,
                            &|v| lcp_bit_lengths[v >> log2_bs],
                            max_lcp,
                            &lcp_counts,
                            VBuilder::default(),
                            pl,
                        )?;

                        // -- lcp2bucket --
                        pl.info(format_args!(
                            "Building LCP prefix → bucket map ({num_buckets} buckets)..."
                        ));
                        let lcp2bucket = VBuilder::<
                            _,
                            BitFieldVec<Box<[usize]>>,
                            [u64; 1],
                            Fuse3NoShards,
                        >::default()
                        .expected_num_keys(num_buckets)
                        .try_build_func::<IntBitPrefix<T>, IntBitPrefix<T>>(
                            FromCloneableIntoIterator::new((0..num_buckets).map(|b| {
                                IntBitPrefix::new(bucket_first_keys[b] ^ T::MIN, lcp_bit_lengths[b])
                            })),
                            FromCloneableIntoIterator::new(0..num_buckets),
                            pl,
                        )?;

                        let off_bits = offsets.data.mem_size(SizeFlags::default()) * 8;
                        let lcp_bits_total =
                            (lcp_lengths.short.data.mem_size(SizeFlags::default())
                                + lcp_lengths.long.data.mem_size(SizeFlags::default())
                                + lcp_lengths.remap.len() * std::mem::size_of::<usize>())
                                * 8;
                        let l2b_bits = lcp2bucket.data.mem_size(SizeFlags::default()) * 8;
                        let total = off_bits + lcp_bits_total + l2b_bits;
                        pl.info(format_args!(
                            "Actual bit cost per key: {:.2} ({total} bits for {n} keys)",
                            total as f64 / n as f64
                        ));

                        Ok(Self {
                            n,
                            log2_bucket_size: log2_bs,
                            offsets,
                            lcp_lengths,
                            lcp2bucket,
                        })
                    }};
                }

                match store {
                    AnyShardStore::Online(mut s) => build_from_store!(&mut s),
                    AnyShardStore::Offline(mut s) => build_from_store!(&mut s),
                }
            },
            pl,
        )
    }
}

// ── Byte-sequence variant ───────────────────────────────────────────

/// A two-step monotone minimal perfect hash function for sorted
/// byte-sequence keys.
///
/// See [`Lcp2MmphfStr`] and [`Lcp2MmphfSliceU8`] for common instantiations.
#[derive(Debug, MemDbg, MemSize)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Lcp2Mmphf<K: ?Sized, S: Sig = [u64; 2], E: ShardEdge<S, 3> = FuseLge3Shards> {
    pub(crate) n: usize,
    pub(crate) log2_bucket_size: usize,
    pub(crate) offsets: VFunc<K, usize, BitFieldVec<Box<[usize]>>, S, E>,
    pub(crate) lcp_lengths: VFunc2<K, usize, BitFieldVec<Box<[usize]>>, S, E, Fuse3Shards>,
    pub(crate) lcp2bucket:
        VFunc<BitPrefix, usize, BitFieldVec<Box<[usize]>>, [u64; 1], Fuse3NoShards>,
}

pub type Lcp2MmphfStr<S = [u64; 2], E = FuseLge3Shards> = Lcp2Mmphf<str, S, E>;
pub type Lcp2MmphfSliceU8<S = [u64; 2], E = FuseLge3Shards> = Lcp2Mmphf<[u8], S, E>;

impl<K: ?Sized + AsRef<[u8]> + ToSig<S>, S: Sig, E: ShardEdge<S, 3>> Lcp2Mmphf<K, S, E>
where
    Fuse3Shards: ShardEdge<S, 3>,
{
    #[inline]
    pub fn get(&self, key: &K) -> usize {
        let sig = K::to_sig(key, self.offsets.seed);
        let offset = self.offsets.get_by_sig_unaligned(sig);
        let lcp_bit_length = self.lcp_lengths.get_by_sig_unaligned(sig);

        let key_bytes: &[u8] = key.as_ref();
        let lcp2b_seed = self.lcp2bucket.seed;
        let lcp2b_sig = if lcp_bit_length <= key_bytes.len() * 8 {
            bit_prefix_sig(key_bytes, lcp_bit_length, lcp2b_seed)
        } else {
            // Rare: LCP extends into the virtual NUL (at most 8 extra bits).
            let mut hasher = xxh3::Xxh3::with_seed(lcp2b_seed);
            hasher.update(key_bytes);
            hasher.update(&[0u8]);
            hasher.update(&lcp_bit_length.to_ne_bytes());
            [hasher.digest()]
        };
        let bucket = self.lcp2bucket.get_by_sig_unaligned(lcp2b_sig);
        (bucket << self.log2_bucket_size) + offset
    }
}

impl<K: ?Sized, S: Sig, E: ShardEdge<S, 3>> Lcp2Mmphf<K, S, E>
where
    Fuse3Shards: ShardEdge<S, 3>,
{
    pub const fn len(&self) -> usize {
        self.n
    }
    pub const fn is_empty(&self) -> bool {
        self.n == 0
    }
}

#[cfg(feature = "rayon")]
impl<K, S, E> Lcp2Mmphf<K, S, E>
where
    K: ?Sized + AsRef<[u8]> + ToSig<S> + std::fmt::Debug,
    S: Sig + Send + Sync,
    E: ShardEdge<S, 3>,
    Fuse3Shards: ShardEdge<S, 3>,
    SigVal<S, usize>: RadixKey,
    SigVal<E::LocalSig, usize>: std::ops::BitXor + std::ops::BitXorAssign,
    SigVal<<Fuse3Shards as ShardEdge<S, 3>>::LocalSig, usize>:
        std::ops::BitXor + std::ops::BitXorAssign,
{
    /// Creates a two-step LCP-based MMPHF for byte-sequence keys
    /// using default [`VBuilder`] settings.
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

    /// Creates a two-step LCP-based MMPHF for byte-sequence keys
    /// using the given [`VBuilder`] configuration.
    ///
    /// The builder controls construction parameters such as offline
    /// mode (`offline`), thread count (`max_num_threads`), sharding
    /// overhead (`eps`), and PRNG seed (`seed`).
    ///
    /// The keys must be in strictly increasing lexicographic order.
    pub fn try_new_with_builder<
        B: ?Sized + AsRef<[u8]> + Borrow<K>,
        P: ProgressLog + Clone + Send + Sync,
    >(
        keys: impl FallibleRewindableLender<
            RewindError: std::error::Error + Send + Sync + 'static,
            Error: std::error::Error + Send + Sync + 'static,
        > + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
        n: usize,
        builder: VBuilder<usize, BitFieldVec<Box<[usize]>>, S, E>,
        pl: &mut P,
    ) -> Result<Self> {
        if n == 0 {
            return Ok(Self {
                n: 0,
                log2_bucket_size: 0,
                offsets: VFunc::empty(),
                lcp_lengths: VFunc2::empty(),
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

        // LCP state shared between key_to_val and build_fn via RefCell.
        // (lcp_bit_lengths, bucket_first_keys, prev_key, curr_lcp_bits)
        let state = RefCell::new((
            Vec::<usize>::with_capacity(num_buckets),
            Vec::<Vec<u8>>::with_capacity(num_buckets),
            Vec::<u8>::new(),
            0usize,
        ));

        let mut builder = builder.expected_num_keys(n);
        builder.try_populate_and_build_with_fn(
            keys,
            &mut |key: &B, idx: usize| -> anyhow::Result<usize> {
                let mut s = state.borrow_mut();
                let (lcp_bit_lengths, bucket_first_keys, prev_key, curr_lcp_bits) = &mut *s;
                let key_bytes: &[u8] = key.as_ref();

                if idx == 0 {
                    lcp_bit_lengths.clear();
                    bucket_first_keys.clear();
                    prev_key.clear();
                    *curr_lcp_bits = 0;
                }

                if idx > 0 && key_bytes <= prev_key.as_slice() {
                    bail!(
                        "Keys are not in strictly increasing lexicographic order \
                         at position {idx}"
                    );
                }

                let offset = idx & bucket_mask;
                if offset == 0 {
                    if idx > 0 {
                        lcp_bit_lengths.push(*curr_lcp_bits);
                    }
                    bucket_first_keys.push(key_bytes.to_vec());
                    *curr_lcp_bits = (key_bytes.len() + 1) * 8;
                } else {
                    *curr_lcp_bits =
                        (*curr_lcp_bits).min(lcp_bits_nul(key_bytes, prev_key));
                }
                prev_key.clear();
                prev_key.extend_from_slice(key_bytes);
                Ok(idx)
            },
            &mut |_builder, seed, store, _max_value, _num_keys, pl: &mut P| {
                let shard_edge = _builder.shard_edge;
                // Finalize LCP data (last bucket).
                {
                    let mut s = state.borrow_mut();
                    let last_lcp = s.3;
                    s.0.push(last_lcp);
                }
                let s = state.borrow();
                let (lcp_bit_lengths, bucket_first_keys, _, _) = &*s;
                assert_eq!(lcp_bit_lengths.len(), num_buckets);

                macro_rules! build_from_store {
                    ($store:expr) => {{
                        let store = $store;
                        pl.info(format_args!(
                            "Building key → offset map ({log2_bs} bits)..."
                        ));
                        let offsets = VBuilder::<usize, BitFieldVec<Box<[usize]>>, S, E>::default()
                            .try_build_func_with_store::<K, usize>(
                                seed,
                                shard_edge,
                                bucket_mask,
                                store,
                                &|_, sig_val| sig_val.val & bucket_mask,
                                &|_| {},
                                pl,
                            )?;

                        // Compute frequency table directly from lcp_bit_lengths
                        // (no store scan needed).
                        let max_lcp = *lcp_bit_lengths.iter().max().unwrap_or(&0);
                        let mut lcp_counts: std::collections::HashMap<usize, usize> =
                            std::collections::HashMap::new();
                        let last_bucket_size = n - (num_buckets - 1) * bucket_size;
                        for (b, &lcp) in lcp_bit_lengths.iter().enumerate() {
                            let bsize = if b == num_buckets - 1 {
                                last_bucket_size
                            } else {
                                bucket_size
                            };
                            *lcp_counts.entry(lcp).or_insert(0) += bsize;
                        }

                        pl.info(format_args!("Building two-step LCP lengths..."));
                        let lcp_lengths = VFunc2::try_build_from_store_with_freq::<usize>(
                            seed,
                            shard_edge,
                            store,
                            &|v| lcp_bit_lengths[v >> log2_bs],
                            max_lcp,
                            &lcp_counts,
                            VBuilder::default(),
                            pl,
                        )?;

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

                        let lcp2bucket = VBuilder::<
                            _,
                            BitFieldVec<Box<[usize]>>,
                            [u64; 1],
                            Fuse3NoShards,
                        >::default()
                        .expected_num_keys(num_buckets)
                        .try_build_func::<BitPrefix, BitPrefix>(
                            FromCloneableIntoIterator::new((0..num_buckets).map(|b| {
                                BitPrefix::new(&extended_first_keys[b], lcp_bit_lengths[b])
                            })),
                            FromCloneableIntoIterator::new(0..num_buckets),
                            pl,
                        )?;

                        let off_bits = offsets.data.mem_size(SizeFlags::default()) * 8;
                        let lcp_bits_total =
                            (lcp_lengths.short.data.mem_size(SizeFlags::default())
                                + lcp_lengths.long.data.mem_size(SizeFlags::default())
                                + lcp_lengths.remap.len() * std::mem::size_of::<usize>())
                                * 8;
                        let l2b_bits = lcp2bucket.data.mem_size(SizeFlags::default()) * 8;
                        let total = off_bits + lcp_bits_total + l2b_bits;
                        pl.info(format_args!(
                            "Actual bit cost per key: {:.2} ({total} bits for {n} keys)",
                            total as f64 / n as f64
                        ));

                        Ok(Self {
                            n,
                            log2_bucket_size: log2_bs,
                            offsets,
                            lcp_lengths,
                            lcp2bucket,
                        })
                    }};
                }

                match store {
                    AnyShardStore::Online(mut s) => build_from_store!(&mut s),
                    AnyShardStore::Offline(mut s) => build_from_store!(&mut s),
                }
            },
            pl,
        )
    }
}
