/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

#![allow(clippy::type_complexity)]

//! Two-step LCP-based monotone minimal perfect hash functions.
//!
//! Compared to [`LcpMmphfInt`](super::LcpMmphfInt) / [`LcpMmphf`](super::LcpMmphf),
//! these variants use a [`VFunc2`] for the LCP-length component, trading
//! ≈3 extra random memory accesses per query for ≈20–35% less space.
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
use crate::func::lcp_mmphf::{BitPrefix, IntBitPrefix, MAGIC_COOKIE, bit_prefix_sig};
use crate::func::shard_edge::{Fuse3NoShards, FuseLge3Shards, ShardEdge};
use crate::func::vfunc2::VFunc2;
use crate::utils::*;
use mem_dbg::*;
use num_primitive::PrimitiveInteger;
use xxhash_rust::xxh3;

#[cfg(feature = "rayon")]
use {
    crate::func::VBuilder,
    crate::func::lcp_mmphf::{lcp_bits, lcp_bits_with_cookie, log2_bucket_size},
    anyhow::{Result, bail},
    dsi_progress_logger::ProgressLog,
    lender::*,
    rdst::RadixKey,
    std::borrow::Borrow,
    std::convert::Infallible,
};

// ── Integer variant ─────────────────────────────────────────────────

/// A two-step monotone minimal perfect hash function for sorted integers.
///
/// Like [`LcpMmphfInt`](super::LcpMmphfInt) but uses a [`VFunc2`] for
/// the LCP-length component, trading extra random reads for less space.
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
///     Lcp2MmphfInt::new(FromSlice::new(&keys), keys.len(), no_logging![])?;
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
pub struct Lcp2MmphfInt<T: PrimitiveInteger, S: Sig = [u64; 2], E: ShardEdge<S, 3> = FuseLge3Shards>
{
    pub(crate) n: usize,
    pub(crate) log2_bucket_size: usize,
    /// Maps each key to its offset within the bucket.
    pub(crate) offsets: VFunc<T, usize, BitFieldVec<Box<[usize]>>, S, E>,
    /// Two-step retrieval of LCP lengths.
    pub(crate) lcp_lengths: VFunc2<T, S, E>,
    /// Maps each LCP bit-prefix to its bucket index.
    pub(crate) lcp2bucket:
        VFunc<IntBitPrefix<T>, usize, BitFieldVec<Box<[usize]>>, [u64; 1], Fuse3NoShards>,
}

impl<T: PrimitiveInteger + ToSig<S>, S: Sig, E: ShardEdge<S, 3>> Lcp2MmphfInt<T, S, E>
where
    Fuse3NoShards: ShardEdge<S, 3>,
{
    #[inline]
    pub fn get(&self, key: T) -> usize
    where
        T: Copy,
    {
        let sig = T::to_sig(key, self.offsets.seed);
        let offset = self.offsets.get_by_sig(sig);
        let lcp_bit_length = self.lcp_lengths.get_by_sig(sig);
        let prefix = IntBitPrefix::new(key ^ T::MIN, lcp_bit_length);
        let bucket = self.lcp2bucket.get(prefix);
        (bucket << self.log2_bucket_size) + offset
    }
}

impl<T: PrimitiveInteger, S: Sig, E: ShardEdge<S, 3>> Lcp2MmphfInt<T, S, E> {
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
    Fuse3NoShards: ShardEdge<S, 3>,
    SigVal<S, usize>: RadixKey,
    SigVal<E::LocalSig, usize>: std::ops::BitXor + std::ops::BitXorAssign,
    SigVal<<Fuse3NoShards as ShardEdge<S, 3>>::LocalSig, usize>:
        std::ops::BitXor + std::ops::BitXorAssign,
{
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

    pub fn new_with_builder(
        mut keys: impl FallibleRewindableLender<
            RewindError: std::error::Error + Send + Sync + 'static,
            Error: std::error::Error + Send + Sync + 'static,
        > + for<'lend> FallibleLending<'lend, Lend = &'lend T>,
        n: usize,
        builder: VBuilder<usize, BitFieldVec<Box<[usize]>>, S, E>,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> Result<Self> {
        if n == 0 {
            let ek: Vec<T> = vec![];
            let ebp: Vec<IntBitPrefix<T>> = vec![];
            let ev: Vec<usize> = vec![];
            let offsets = VBuilder::<_, BitFieldVec<Box<[usize]>>, S, E>::default()
                .try_build_func::<T, T>(FromSlice::new(&ek), FromSlice::new(&ev), pl)?;
            let lcp_lengths = VFunc2 {
                short: None,
                long: VBuilder::<_, BitFieldVec<Box<[usize]>>, S, Fuse3NoShards>::default()
                    .try_build_func::<T, T>(FromSlice::new(&ek), FromSlice::new(&ev), pl)?,
                remap: Box::new([]),
                escape: 0,
            };
            let lcp2bucket =
                VBuilder::<_, BitFieldVec<Box<[usize]>>, [u64; 1], Fuse3NoShards>::default()
                    .try_build_func::<IntBitPrefix<T>, IntBitPrefix<T>>(
                    FromSlice::new(&ebp),
                    FromSlice::new(&ev),
                    pl,
                )?;
            return Ok(Self {
                n: 0,
                log2_bucket_size: 0,
                offsets,
                lcp_lengths,
                lcp2bucket,
            });
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
                        "Keys are not in strictly increasing order at position {i}: {prev:?} >= {key:?}"
                    );
                }
            }
            let offset = i & bucket_mask;
            if offset == 0 {
                if i > 0 {
                    lcp_bit_lengths.push(curr_lcp_bits);
                }
                bucket_first_keys.push(key);
                curr_lcp_bits = T::BITS as usize;
            } else {
                curr_lcp_bits = curr_lcp_bits.min(lcp_bits(key, prev_key.unwrap()));
            }
            prev_key = Some(key);
            i += 1;
        }
        assert_eq!(i, n, "Expected {n} keys but got {i}");
        lcp_bit_lengths.push(curr_lcp_bits);

        // -- Build from shared store --
        pl.info(format_args!("Hashing keys..."));
        let keys = keys.rewind()?;

        let (seed_vfunc, store) = builder.expected_num_keys(n)._try_build_func::<T, T>(
            keys,
            FromIntoFallibleLenderFactory::new(|| {
                Ok::<_, Infallible>(FromCloneableIntoIterator::new((0..n).map(|idx| {
                    (lcp_bit_lengths[idx >> log2_bs] << log2_bs) | (idx & bucket_mask)
                })))
            })?,
            true,
            pl,
        )?;

        let seed = seed_vfunc.seed;
        let shard_edge = seed_vfunc.shard_edge;
        let mut store = match store {
            Some(AnyShardStore::Online(s)) => s,
            _ => unreachable!("keep_store=true with online store"),
        };

        // -- Offsets --
        pl.info(format_args!(
            "Building key → offset map ({log2_bs} bits)..."
        ));
        let offsets = VBuilder::<usize, BitFieldVec<Box<[usize]>>, S, E>::default()
            .try_build_func_from_store::<T, usize>(
                seed,
                shard_edge,
                n,
                bucket_mask,
                &mut store,
                &|_, sig_val| sig_val.val & bucket_mask,
                pl,
            )?;

        // -- LCP lengths (two-step) --
        pl.info(format_args!("Building two-step LCP lengths..."));
        let lcp_lengths = VFunc2::try_build_from_store::<usize>(
            seed,
            shard_edge,
            n,
            &mut store,
            &|v| v >> log2_bs,
            pl,
        )?;

        // -- lcp2bucket --
        pl.info(format_args!(
            "Building LCP prefix → bucket map ({num_buckets} buckets)..."
        ));
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

        let off_bits = offsets.data.mem_size(SizeFlags::default()) * 8;
        let lcp_bits_total = (lcp_lengths
            .short
            .as_ref()
            .map_or(0, |f| f.data.mem_size(SizeFlags::default()))
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
    pub(crate) lcp_lengths: VFunc2<K, S, E>,
    pub(crate) lcp2bucket:
        VFunc<BitPrefix, usize, BitFieldVec<Box<[usize]>>, [u64; 1], Fuse3NoShards>,
}

pub type Lcp2MmphfStr<S = [u64; 2], E = FuseLge3Shards> = Lcp2Mmphf<str, S, E>;
pub type Lcp2MmphfSliceU8<S = [u64; 2], E = FuseLge3Shards> = Lcp2Mmphf<[u8], S, E>;

impl<K: ?Sized + AsRef<[u8]> + ToSig<S>, S: Sig, E: ShardEdge<S, 3>> Lcp2Mmphf<K, S, E>
where
    Fuse3NoShards: ShardEdge<S, 3>,
{
    #[inline]
    pub fn get(&self, key: &K) -> usize {
        let sig = K::to_sig(key, self.offsets.seed);
        let offset = self.offsets.get_by_sig(sig);
        let lcp_bit_length = self.lcp_lengths.get_by_sig(sig);

        let key_bytes: &[u8] = key.as_ref();
        let lcp2b_seed = self.lcp2bucket.seed;
        let lcp2b_sig = if lcp_bit_length <= key_bytes.len() * 8 {
            bit_prefix_sig(key_bytes, lcp_bit_length, lcp2b_seed)
        } else {
            let mut hasher = xxh3::Xxh3::with_seed(lcp2b_seed);
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
        let bucket = self.lcp2bucket.get_by_sig(lcp2b_sig);
        (bucket << self.log2_bucket_size) + offset
    }
}

impl<K: ?Sized, S: Sig, E: ShardEdge<S, 3>> Lcp2Mmphf<K, S, E> {
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
    Fuse3NoShards: ShardEdge<S, 3>,
    SigVal<S, usize>: RadixKey,
    SigVal<E::LocalSig, usize>: std::ops::BitXor + std::ops::BitXorAssign,
    SigVal<<Fuse3NoShards as ShardEdge<S, 3>>::LocalSig, usize>:
        std::ops::BitXor + std::ops::BitXorAssign,
{
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

    pub fn new_with_builder<B: ?Sized + AsRef<[u8]> + Borrow<K>>(
        mut keys: impl FallibleRewindableLender<
            RewindError: std::error::Error + Send + Sync + 'static,
            Error: std::error::Error + Send + Sync + 'static,
        > + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
        n: usize,
        builder: VBuilder<usize, BitFieldVec<Box<[usize]>>, S, E>,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> Result<Self> {
        if n == 0 {
            let ek: Vec<&K> = vec![];
            let ebp: Vec<BitPrefix> = vec![];
            let ev: Vec<usize> = vec![];
            let offsets = VBuilder::<_, BitFieldVec<Box<[usize]>>, S, E>::default()
                .try_build_func::<K, &K>(FromSlice::new(&ek), FromSlice::new(&ev), pl)?;
            let lcp_lengths = VFunc2 {
                short: None,
                long: VBuilder::<_, BitFieldVec<Box<[usize]>>, S, Fuse3NoShards>::default()
                    .try_build_func::<K, &K>(FromSlice::new(&ek), FromSlice::new(&ev), pl)?,
                remap: Box::new([]),
                escape: 0,
            };
            let lcp2bucket =
                VBuilder::<_, BitFieldVec<Box<[usize]>>, [u64; 1], Fuse3NoShards>::default()
                    .try_build_func::<BitPrefix, BitPrefix>(
                    FromSlice::new(&ebp),
                    FromSlice::new(&ev),
                    pl,
                )?;
            return Ok(Self {
                n: 0,
                log2_bucket_size: 0,
                offsets,
                lcp_lengths,
                lcp2bucket,
            });
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
        let mut bucket_first_keys: Vec<Vec<u8>> = Vec::with_capacity(num_buckets);
        let mut prev_key: Vec<u8> = Vec::new();
        let mut curr_lcp_bits: usize = 0;
        let mut i = 0usize;

        while let Some(key) = keys.next()? {
            let key_bytes: &[u8] = key.as_ref();
            if i > 0 && key_bytes <= prev_key.as_slice() {
                bail!("Keys are not in strictly increasing lexicographic order at position {i}");
            }
            let offset = i & bucket_mask;
            if offset == 0 {
                if i > 0 {
                    lcp_bit_lengths.push(curr_lcp_bits);
                }
                bucket_first_keys.push(key_bytes.to_vec());
                curr_lcp_bits = (key_bytes.len() + MAGIC_COOKIE.len()) * 8;
            } else {
                curr_lcp_bits = curr_lcp_bits.min(lcp_bits_with_cookie(key_bytes, &prev_key));
            }
            prev_key.clear();
            prev_key.extend_from_slice(key_bytes);
            i += 1;
        }
        assert_eq!(i, n, "Expected {n} keys but got {i}");
        lcp_bit_lengths.push(curr_lcp_bits);

        // -- Build from shared store --
        pl.info(format_args!("Hashing keys..."));
        let keys = keys.rewind()?;

        let (seed_vfunc, store) = builder.expected_num_keys(n)._try_build_func::<K, B>(
            keys,
            FromIntoFallibleLenderFactory::new(|| {
                Ok::<_, Infallible>(FromCloneableIntoIterator::new((0..n).map(|idx| {
                    (lcp_bit_lengths[idx >> log2_bs] << log2_bs) | (idx & bucket_mask)
                })))
            })?,
            true,
            pl,
        )?;

        let seed = seed_vfunc.seed;
        let shard_edge = seed_vfunc.shard_edge;
        let mut store = match store {
            Some(AnyShardStore::Online(s)) => s,
            _ => unreachable!("keep_store=true with online store"),
        };

        pl.info(format_args!(
            "Building key → offset map ({log2_bs} bits)..."
        ));
        let offsets = VBuilder::<usize, BitFieldVec<Box<[usize]>>, S, E>::default()
            .try_build_func_from_store::<K, usize>(
                seed,
                shard_edge,
                n,
                bucket_mask,
                &mut store,
                &|_, sig_val| sig_val.val & bucket_mask,
                pl,
            )?;

        pl.info(format_args!("Building two-step LCP lengths..."));
        let lcp_lengths = VFunc2::try_build_from_store::<usize>(
            seed,
            shard_edge,
            n,
            &mut store,
            &|v| v >> log2_bs,
            pl,
        )?;

        pl.info(format_args!(
            "Building LCP prefix → bucket map ({num_buckets} buckets)..."
        ));
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

        let off_bits = offsets.data.mem_size(SizeFlags::default()) * 8;
        let lcp_bits_total = (lcp_lengths
            .short
            .as_ref()
            .map_or(0, |f| f.data.mem_size(SizeFlags::default()))
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
    }
}
