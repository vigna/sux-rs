/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

#![allow(clippy::type_complexity)]

#[cfg(feature = "rayon")]
use core::error::Error;
use std::borrow::Borrow;

use super::shard_edge::FuseLge3Shards;
use crate::bits::BitFieldVec;
use crate::func::VFunc;
use crate::func::shard_edge::ShardEdge;
use crate::traits::Word;
use crate::utils::*;
use mem_dbg::*;
use value_traits::slices::SliceByValue;

/// A two-step static function that stores frequent values in a narrow first
/// function and infrequent values in a wider second function, with
/// frequency-based remapping.
///
/// During construction, the value distribution is analyzed and the most frequent
/// values are assigned compact indices (0 . . 2*ʳ* − 1). A first ("short")
/// [`VFunc`] maps every key either to one of these compact indices or to the
/// escape sentinel 2*ʳ* − 1. For escaped keys a second ("long") [`VFunc`],
/// built only over the escaped subset, stores the full value. A small `remap`
/// table converts compact indices back to the original values.
///
/// The split point bit width of the short function *r* is chosen to minimize
/// the total estimated space.
///
/// If the distribution of output values is very skewed, this function will use
/// much less space than a [`VFunc`]. The impact on query time is limited to the
/// additional access to the long function; sometimes, the reduction in space
/// can even lead to faster queries due to better cache locality.
///
/// This structure implements the [`TryIntoUnaligned`] trait, allowing it to be
/// converted into (usually faster) structures using unaligned access.
///
/// # Generics
///
/// * `T`: The type of the keys.
/// * `W`: The word used to store the data, which is also the output type. It
///   can be any unsigned type. Defaults to `usize`.
/// * `D`: The backend storing the function data. Defaults to
///   `BitFieldVec<Box<[W]>>`.
/// * `S`: The signature type. The default is `[u64; 2]`.
/// * `E0`: The sharding and edge logic type for the short (frequent-value)
///   function. The default is [`FuseLge3Shards`].
/// * `E1`: The sharding and edge logic type for the long (escaped-value)
///   function. The default is `E0`.
///
/// # References
///
/// Djamal Belazzougui, Paolo Boldi, Rasmus Pagh, and Sebastiano Vigna. [Theory
/// and practice of monotone minimal perfect
/// hashing](https://doi.org/10.1145/1963190.2025378). *ACM Journal of
/// Experimental Algorithmics*, 16(3):3.2:1−3.2:26, 2011.
#[derive(Debug, MemDbg, MemSize)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde))]
#[cfg_attr(
    feature = "epserde",
    epserde(bound(deser = "W: for<'a> epserde::deser::DeserInner<DeserType<'a> = W>"))
)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VFunc2<
    T: ?Sized,
    W = usize,
    D = BitFieldVec<Box<[W]>>,
    S = [u64; 2],
    E0 = FuseLge3Shards,
    E1 = E0,
> {
    /// First function: maps each key to a remapped index (*r* bits), or
    /// `escape` for infrequent values. When *r* = 0 this is an empty
    /// VFunc that always returns 0 = `escape`, so the long function is
    /// always queried.
    pub(crate) short: VFunc<T, W, D, S, E0>,
    /// Second function: maps escaped keys to their full value.
    pub(crate) long: VFunc<T, W, D, S, E1>,
    /// Maps remapped indices (0 . . `escape` − 1) back to actual values.
    pub(crate) remap: Box<[W]>,
    /// The escape value (2*ʳ* − 1). When *r* = 0, `escape` = 0 and the
    /// short function always returns the escape.
    pub(crate) escape: W,
}

impl<T: ?Sized, W: Word, S: Sig, E0: ShardEdge<S, 3>, E1: ShardEdge<S, 3>>
    VFunc2<T, W, BitFieldVec<Box<[W]>>, S, E0, E1>
{
    /// Creates a VFunc2 with zero keys.
    ///
    /// With `escape = 0`, the short function always returns the escape,
    /// so `get` always queries the long function (which returns zero
    /// since both internal [`VFunc`]s are empty).
    pub fn empty() -> Self {
        Self {
            short: VFunc::empty(),
            long: VFunc::empty(),
            remap: Box::new([]),
            escape: W::ZERO,
        }
    }
}

impl<
    T: ?Sized + ToSig<S>,
    W: Word + BinSafe,
    D: SliceByValue<Value = W>,
    S: Sig,
    E0: ShardEdge<S, 3>,
    E1: ShardEdge<S, 3>,
> VFunc2<T, W, D, S, E0, E1>
{
    /// Retrieves the value for a key given its pre-computed signature.
    ///
    /// The signature must have been computed with `self.short.seed`
    /// (i.e., `T::to_sig(key, self.short.seed)`). Both the short and
    /// long internal functions share the same seed.
    ///
    /// This method is mainly useful in the construction of compound
    /// functions.
    #[inline]
    pub fn get_by_sig(&self, sig: S) -> W {
        let idx = self.short.get_by_sig(sig);
        if idx != self.escape {
            self.remap[idx.as_u128() as usize]
        } else {
            self.long.get_by_sig(sig)
        }
    }

    /// Retrieves the value associated with the given key, or an arbitrary
    /// value if the key was not in the original set.
    #[inline(always)]
    pub fn get(&self, key: impl Borrow<T>) -> W {
        self.get_by_sig(T::to_sig(key.borrow(), self.short.seed))
    }
}

// ── Aligned ↔ Unaligned conversions ─────────────────────────────────

use crate::bits::BitFieldVecU;
use crate::traits::TryIntoUnaligned;

impl<T: ?Sized, W: Word, S: Sig, E0: ShardEdge<S, 3>, E1: ShardEdge<S, 3>> TryIntoUnaligned
    for VFunc2<T, W, BitFieldVec<Box<[W]>>, S, E0, E1>
{
    type Unaligned = VFunc2<T, W, BitFieldVecU<Box<[W]>>, S, E0, E1>;
    fn try_into_unaligned(
        self,
    ) -> Result<Self::Unaligned, crate::traits::UnalignedConversionError> {
        Ok(VFunc2 {
            short: self.short.try_into_unaligned()?,
            long: self.long.try_into_unaligned()?,
            remap: self.remap,
            escape: self.escape,
        })
    }
}

impl<T: ?Sized, W: Word, S: Sig, E0: ShardEdge<S, 3>, E1: ShardEdge<S, 3>>
    From<VFunc2<T, W, BitFieldVecU<Box<[W]>>, S, E0, E1>>
    for VFunc2<T, W, BitFieldVec<Box<[W]>>, S, E0, E1>
{
    /// Converts a [`VFunc2`] with [`BitFieldVecU`] data back into
    /// one with [`BitFieldVec`] data, removing the padding words.
    fn from(vf: VFunc2<T, W, BitFieldVecU<Box<[W]>>, S, E0, E1>) -> Self {
        VFunc2 {
            short: VFunc::from(vf.short),
            long: VFunc::from(vf.long),
            remap: vf.remap,
            escape: vf.escape,
        }
    }
}

/// Finds the optimal short-function bit width `r` for a [`VFunc2`],
/// minimizing the estimated total space.
///
/// `sorted_vals` must be the distinct values sorted by descending
/// frequency.
#[cfg(feature = "rayon")]
fn find_optimal_r<W: Word>(
    n: usize,
    max_value: usize,
    sorted_vals: &[W],
    counts: &std::collections::HashMap<W, usize>,
    w_bits: usize,
) -> usize {
    let w = (max_value as u128).bit_len() as usize;
    let m = sorted_vals.len();
    let c = 1.11f64; // VFunc expansion factor (approximate)

    let mut post = n;
    let mut pos = 0usize;
    let mut best_r = 0usize;
    let mut best_cost = f64::MAX;

    // r < w <= 64, so 1usize << r never overflows.
    for r in 0..w {
        let cost_first = if r == 0 { 0.0 } else { c * n as f64 * r as f64 };
        let cost_second = c * post as f64 * w as f64;
        let cost_remap = pos as f64 * w_bits as f64;
        let cost = cost_first + cost_second + cost_remap;

        if cost < best_cost {
            best_cost = cost;
            best_r = r;
        }

        let to_absorb = (1usize << r).min(m - pos);
        for _ in 0..to_absorb {
            post -= counts[&sorted_vals[pos]];
            pos += 1;
        }
    }

    best_r
}

#[cfg(feature = "rayon")]
use {
    crate::func::VBuilder,
    dsi_progress_logger::ProgressLog,
    lender::*,
    rdst::RadixKey,
    std::ops::{BitXor, BitXorAssign},
    sync_cell_slice::SyncSlice,
};

#[cfg(feature = "rayon")]
impl<T, W, S, E0, E1> VFunc2<T, W, BitFieldVec<Box<[W]>>, S, E0, E1>
where
    T: ?Sized + ToSig<S> + std::fmt::Debug,
    W: Word + BinSafe,
    S: Sig + Send + Sync,
    E0: ShardEdge<S, 3>,
    E1: ShardEdge<S, 3>,
    SigVal<S, W>: RadixKey,
    SigVal<E0::LocalSig, W>: BitXor + BitXorAssign,
    SigVal<E1::LocalSig, W>: BitXor + BitXorAssign,
{
    /// Builds a [`VFunc2`] from keys and values using default
    /// [`VBuilder`] settings.
    ///
    /// * `keys` and `values` must be aligned (one value per key, same
    ///   order) and rewindable (they may be rewound on retry).
    /// * `n` is the expected number of keys; a significantly wrong
    ///   value may degrade performance or cause extra retries.
    ///
    /// This is a convenience wrapper around
    /// [`try_new_with_builder`](Self::try_new_with_builder) with
    /// `VBuilder::default()`.
    pub fn try_new<B: ?Sized + std::borrow::Borrow<T>>(
        keys: impl FallibleRewindableLender<
            RewindError: Error + Send + Sync + 'static,
            Error: Error + Send + Sync + 'static,
        > + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
        values: impl FallibleRewindableLender<
            RewindError: Error + Send + Sync + 'static,
            Error: Error + Send + Sync + 'static,
        > + for<'lend> FallibleLending<'lend, Lend = &'lend W>,
        n: usize,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> anyhow::Result<Self> {
        Self::try_new_with_builder(keys, values, n, VBuilder::default(), pl)
    }

    /// Builds a [`VFunc2`] from keys and values using the given
    /// [`VBuilder`] configuration.
    ///
    /// * `keys` and `values` must be aligned (one value per key, same
    ///   order) and rewindable (they may be rewound on retry).
    /// * `n` is the expected number of keys.
    ///
    /// The builder controls construction parameters such as offline
    /// mode (`offline`), thread count (`max_num_threads`), sharding
    /// overhead (`eps`), and PRNG seed (`seed`).
    pub fn try_new_with_builder<B: ?Sized + std::borrow::Borrow<T>>(
        keys: impl FallibleRewindableLender<
            RewindError: Error + Send + Sync + 'static,
            Error: Error + Send + Sync + 'static,
        > + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
        values: impl FallibleRewindableLender<
            RewindError: Error + Send + Sync + 'static,
            Error: Error + Send + Sync + 'static,
        > + for<'lend> FallibleLending<'lend, Lend = &'lend W>,
        n: usize,
        builder: VBuilder<W, BitFieldVec<Box<[W]>>, S, E0>,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> anyhow::Result<Self> {
        let mut builder = builder.expected_num_keys(n);
        builder.try_populate_and_build(
            keys,
            values,
            &mut |builder, seed, store, _max_value, _num_keys, pl| match store {
                AnyShardStore::Online(mut s) => Self::try_build_from_store::<W>(
                    seed,
                    builder.shard_edge,
                    &mut s,
                    &|v| v,
                    VBuilder::default()
                        .max_num_threads(builder.max_num_threads)
                        .eps(builder.eps),
                    pl,
                ),
                AnyShardStore::Offline(mut s) => Self::try_build_from_store::<W>(
                    seed,
                    builder.shard_edge,
                    &mut s,
                    &|v| v,
                    VBuilder::default()
                        .max_num_threads(builder.max_num_threads)
                        .eps(builder.eps),
                    pl,
                ),
            },
            pl,
        )
    }

    /// Builds a [`VFunc2`] from an existing [`ShardStore`].
    ///
    /// This is the low-level constructor used when multiple VFuncs are
    /// built from the same store (e.g., inside
    /// [`Lcp2Mmphf`](crate::func::Lcp2Mmphf)). Use [`try_new`](Self::try_new)
    /// or [`try_new_with_builder`](Self::try_new_with_builder) for the
    /// common case of building from keys and values directly.
    ///
    /// # Preconditions
    ///
    /// * `seed` and `shard_edge` must be the values used when the store
    ///   was populated.
    /// * `get_val` must be deterministic: the store is iterated multiple
    ///   times (frequency analysis, then short/long construction) and
    ///   differing results corrupt the function silently.
    ///
    /// # Arguments
    ///
    /// * `seed` — the seed from the store's population step.
    /// * `shard_edge` — the shard edge from the same population step.
    /// * `store` — the populated shard store.
    /// * `get_val` — extracts the value from the store's packed entry
    ///   (e.g., `|v| v >> log2_bs` for LCP lengths).
    /// * `builder` — the builder configuration for the internal VFuncs.
    /// * `pl` — a progress logger.
    pub fn try_build_from_store<V: BinSafe + Default + Send + Sync + Copy>(
        seed: u64,
        shard_edge: E0,
        store: &mut impl ShardStore<S, V>,
        get_val: &(impl Fn(V) -> W + Send + Sync),
        builder: VBuilder<W, BitFieldVec<Box<[W]>>, S, E0>,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> anyhow::Result<Self>
    where
        SigVal<S, V>: RadixKey,
        SigVal<E0::LocalSig, V>: BitXor + BitXorAssign,
        SigVal<E1::LocalSig, V>: BitXor + BitXorAssign,
    {
        // -- 1. Frequency analysis --

        let mut counts: std::collections::HashMap<W, usize> = std::collections::HashMap::new();
        let mut max_value = W::ZERO;

        for shard in store.iter() {
            for sv in shard.iter() {
                let val = get_val(sv.val);
                *counts.entry(val).or_insert(0) += 1;
                if val > max_value {
                    max_value = val;
                }
            }
        }

        Self::try_build_from_store_with_freq(
            seed, shard_edge, store, get_val, max_value, &counts, builder, pl,
        )
    }

    /// Like [`try_build_from_store`](Self::try_build_from_store), but
    /// the caller provides pre-computed value frequencies, avoiding
    /// a full scan of the store for frequency analysis.
    ///
    /// This is a low-level method that requires a thorough understanding
    /// of the builder's internal state.
    ///
    /// # Arguments
    ///
    /// * `max_value` — the maximum value returned by `get_val`.
    /// * `counts` — maps each distinct value to its frequency (number
    ///   of keys with that value). Must be consistent with what `get_val`
    ///   would produce when applied to the store.
    pub fn try_build_from_store_with_freq<V: BinSafe + Default + Send + Sync + Copy>(
        seed: u64,
        shard_edge: E0,
        store: &mut impl ShardStore<S, V>,
        get_val: &(impl Fn(V) -> W + Send + Sync),
        max_value: W,
        counts: &std::collections::HashMap<W, usize>,
        builder: VBuilder<W, BitFieldVec<Box<[W]>>, S, E0>,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> anyhow::Result<Self>
    where
        SigVal<S, V>: RadixKey,
        SigVal<E0::LocalSig, V>: BitXor + BitXorAssign,
        SigVal<E1::LocalSig, V>: BitXor + BitXorAssign,
    {
        let max_value_usize = max_value.as_u128() as usize;

        // -- Sort distinct values by descending frequency --

        let mut sorted_vals: Vec<W> = counts.keys().copied().collect();
        sorted_vals.sort_by(|a, b| counts[b].cmp(&counts[a]));

        let w = max_value.as_u128().bit_len() as usize;
        let m = sorted_vals.len();

        // -- Find optimal r --

        let n = store.len();
        let best_r = find_optimal_r(n, max_value_usize, &sorted_vals, counts, W::BITS as usize);

        let escape_usize = (1usize << best_r).wrapping_sub(1); // 2^r - 1
        let escape = W::try_from(escape_usize).ok().unwrap();
        let num_remapped = escape_usize.min(m);

        // -- Build remap and inv_map --

        let remap: Box<[W]> = sorted_vals[..num_remapped].to_vec().into_boxed_slice();
        let mut inv_map: std::collections::HashMap<W, W> =
            std::collections::HashMap::with_capacity(num_remapped);
        for (i, &val) in remap.iter().enumerate() {
            inv_map.insert(val, W::try_from(i).ok().unwrap());
        }

        pl.info(format_args!(
            "Two-step: r={best_r}, escape={escape_usize}, {num_remapped} remapped values, \
             {m} distinct values, max_value={max_value_usize} ({w} bits)"
        ));

        // -- Build short VFunc --
        // When r = 0, escape = 0 and the short function maps every key to
        // 0 = escape, so the long function is always queried.

        // Set up per-shard counters for escaped entries, using the
        // store's current shard granularity.
        let shb = shard_edge.shard_high_bits();
        let num_shards = 1usize << shb;
        let shard_mask = (1u64 << shb) - 1;
        let mut escaped_counts = vec![0usize; num_shards];
        let sync_counts = escaped_counts.as_sync_slice();

        // Save builder settings before the short VFunc consumes it.
        let saved_max_num_threads = builder.max_num_threads;
        let saved_eps = builder.eps;

        pl.info(format_args!(
            "Building key -> remapped index ({best_r} bits, escape={escape_usize})..."
        ));
        let short = builder.try_build_func_with_store::<T, V>(
            seed,
            shard_edge,
            escape,
            store,
            &|_e, sig_val| {
                let val = get_val(sig_val.val);
                inv_map.get(&val).copied().unwrap_or(escape)
            },
            &|sv: &SigVal<S, V>| {
                if !inv_map.contains_key(&get_val(sv.val)) {
                    let shard_idx = sv.sig.high_bits(shb, shard_mask) as usize;
                    // SAFETY: each shard is processed by exactly one
                    // thread, so no two threads access the same counter.
                    unsafe {
                        let c = sync_counts[shard_idx].get();
                        sync_counts[shard_idx].set(c + 1);
                    }
                }
            },
            pl,
        )?;

        // escaped_counts now has per-shard escaped entry counts.

        // -- Build long VFunc (escaped keys only) --

        let n_escaped = n - sorted_vals[..num_remapped]
            .iter()
            .map(|v| counts[v])
            .sum::<usize>();

        debug_assert_eq!(
            escaped_counts.iter().sum::<usize>(),
            n_escaped,
            "inspect-counted escaped != freq-computed escaped"
        );

        let mut long_shard_edge = E1::default();
        long_shard_edge.set_up_shards(n_escaped, saved_eps);
        let long_shard_high_bits = long_shard_edge.shard_high_bits();

        // Aggregate escaped_counts to the long function's shard granularity.
        let long_num_shards = 1usize << long_shard_high_bits;
        let filtered_shard_sizes = if long_num_shards >= num_shards {
            escaped_counts
        } else {
            let shards_per_long = num_shards / long_num_shards;
            escaped_counts
                .chunks(shards_per_long)
                .map(|chunk| chunk.iter().sum())
                .collect()
        };

        pl.info(format_args!(
            "Building key -> full value ({w} bits, {n_escaped} escaped keys, {:.1}%)...",
            100.0 * n_escaped as f64 / n as f64
        ));

        let mut filtered_store = FilteredShardStore::new(
            store,
            long_shard_high_bits,
            |sv: &SigVal<S, V>| !inv_map.contains_key(&get_val(sv.val)),
            filtered_shard_sizes,
        );
        let long = VBuilder::<W, BitFieldVec<Box<[W]>>, S, E1>::default()
            .max_num_threads(saved_max_num_threads)
            .try_build_func_with_store::<T, V>(
                seed,
                long_shard_edge,
                max_value,
                &mut filtered_store,
                &|_e, sig_val| get_val(sig_val.val),
                &|_| {},
                pl,
            )?;

        Ok(Self {
            short,
            long,
            remap,
            escape,
        })
    }
}
