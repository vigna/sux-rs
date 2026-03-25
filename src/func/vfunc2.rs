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
use crate::func::shard_edge::{Fuse3Shards, ShardEdge};
use crate::utils::*;
use mem_dbg::*;

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
/// # References
///
/// Djamal Belazzougui, Paolo Boldi, Rasmus Pagh, and Sebastiano Vigna. [Theory
/// and practice of monotone minimal perfect
/// hashing](https://doi.org/10.1145/1963190.2025378). *ACM Journal of
/// Experimental Algorithmics*, 16(3):3.2:1−3.2:26, 2011.
#[derive(Debug, MemDbg, MemSize)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VFunc2<T: ?Sized, S: Sig = [u64; 2], E: ShardEdge<S, 3> = FuseLge3Shards> {
    /// First function: maps each key to a remapped index (*r* bits), or
    /// `escape` for infrequent values. When *r* = 0 this is an empty
    /// VFunc that always returns 0 = `escape`, so the long function is
    /// always queried.
    pub(crate) short: VFunc<T, usize, BitFieldVec<Box<[usize]>>, S, E>,
    /// Second function: maps escaped keys to their full value.
    /// Uses Fuse3Shards so we do not have problems with a small number of
    /// escaped keys.
    pub(crate) long: VFunc<T, usize, BitFieldVec<Box<[usize]>>, S, Fuse3Shards>,
    /// Maps remapped indices (0 . . `escape` − 1) back to actual values.
    pub(crate) remap: Box<[usize]>,
    /// The escape value (2*ʳ* − 1). When *r* = 0, `escape` = 0 and the
    /// short function always returns the escape.
    pub(crate) escape: usize,
}

impl<T: ?Sized, S: Sig, E: ShardEdge<S, 3>> VFunc2<T, S, E>
where
    Fuse3Shards: ShardEdge<S, 3>,
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
            escape: 0,
        }
    }
}

impl<T: ?Sized + ToSig<S>, S: Sig, E: ShardEdge<S, 3>> VFunc2<T, S, E>
where
    Fuse3Shards: ShardEdge<S, 3>,
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
    pub fn get_by_sig(&self, sig: S) -> usize {
        let idx = self.short.get_by_sig(sig);
        if idx != self.escape {
            self.remap[idx]
        } else {
            self.long.get_by_sig(sig)
        }
    }

    /// Retrieves the value associated with the given key, or an arbitrary
    /// value if the key was not in the original set.
    #[inline(always)]
    pub fn get(&self, key: impl Borrow<T>) -> usize
    where
        E: ShardEdge<S, 3>,
    {
        self.get_by_sig(T::to_sig(key.borrow(), self.short.seed))
    }
}

#[cfg(feature = "rayon")]
use {
    crate::func::VBuilder,
    dsi_progress_logger::ProgressLog,
    lender::*,
    rdst::RadixKey,
    std::ops::{BitXor, BitXorAssign},
};

#[cfg(feature = "rayon")]
impl<T, S, E> VFunc2<T, S, E>
where
    T: ?Sized + ToSig<S> + std::fmt::Debug,
    S: Sig + Send + Sync,
    E: ShardEdge<S, 3>,
    Fuse3Shards: ShardEdge<S, 3>,
    SigVal<S, usize>: RadixKey,
    SigVal<E::LocalSig, usize>: BitXor + BitXorAssign,
    SigVal<<Fuse3Shards as ShardEdge<S, 3>>::LocalSig, usize>: BitXor + BitXorAssign,
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
        > + for<'lend> FallibleLending<'lend, Lend = &'lend usize>,
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
        > + for<'lend> FallibleLending<'lend, Lend = &'lend usize>,
        n: usize,
        builder: VBuilder<usize, BitFieldVec<Box<[usize]>>, S, E>,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> anyhow::Result<Self> {
        let mut builder = builder.expected_num_keys(n);
        builder.try_populate_and_build(
            keys,
            values,
            &mut |builder, seed, store, _max_value, _num_keys, pl| {
                let mut store = match store {
                    AnyShardStore::Online(s) => s,
                    _ => unreachable!("online builder"),
                };
                Self::try_build_from_store::<usize>(
                    seed,
                    builder.shard_edge,
                    n,
                    &mut store,
                    &|v| v,
                    pl,
                )
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
    /// * `n` — total number of keys in the store.
    /// * `store` — the populated shard store.
    /// * `get_val` — extracts the value from the store's packed entry
    ///   (e.g., `|v| v >> log2_bs` for LCP lengths).
    /// * `pl` — a progress logger.
    pub fn try_build_from_store<V: BinSafe + Default + Send + Sync + Copy>(
        seed: u64,
        shard_edge: E,
        n: usize,
        store: &mut impl ShardStore<S, V>,
        get_val: &(impl Fn(V) -> usize + Send + Sync),
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> anyhow::Result<Self>
    where
        SigVal<S, V>: RadixKey,
        SigVal<E::LocalSig, V>: BitXor + BitXorAssign,
        SigVal<<Fuse3Shards as ShardEdge<S, 3>>::LocalSig, V>: BitXor + BitXorAssign,
    {
        // -- 1. Frequency analysis (array-based) --

        let mut max_value = 0usize;
        for shard in store.iter() {
            for sv in shard.iter() {
                max_value = max_value.max(get_val(sv.val));
            }
        }

        let mut counts = vec![0usize; max_value + 1];
        for shard in store.iter() {
            for sv in shard.iter() {
                counts[get_val(sv.val)] += 1;
            }
        }

        // -- 2. Sort distinct values by descending frequency --

        let mut sorted_vals: Vec<usize> = (0..=max_value).filter(|&v| counts[v] > 0).collect();
        sorted_vals.sort_by(|&a, &b| counts[b].cmp(&counts[a]));

        let w = (max_value as u128).bit_len() as usize;
        let m = sorted_vals.len(); // number of distinct values
        let c = 1.11f64; // VFunc expansion factor (approximate)

        // -- 3. Find optimal r exhaustively --

        let mut post = n; // keys not yet covered by the short function
        let mut pos = 0usize; // position in sorted_vals
        let mut best_r = 0usize;
        let mut best_cost = f64::MAX;

        for r in 0..w {
            // cost(r) = C * n * r + C * n_escaped * w + escape * 64
            let cost_first = if r == 0 { 0.0 } else { c * n as f64 * r as f64 };
            let cost_second = c * post as f64 * w as f64;
            let cost_remap = pos as f64 * 64.0;
            let cost = cost_first + cost_second + cost_remap;

            if cost < best_cost {
                best_cost = cost;
                best_r = r;
            }

            // Absorb the next 2^r values (indices pos..pos+2^r) into the
            // short function.
            let to_absorb = (1usize << r).min(m - pos);
            for _ in 0..to_absorb {
                if pos >= m {
                    break;
                }
                post -= counts[sorted_vals[pos]];
                pos += 1;
            }
        }

        // Cap r to avoid huge remap arrays.
        if best_r >= usize::BITS as usize {
            best_r = usize::BITS as usize - 1;
        }

        Self::try_build_from_store_with_r(seed, shard_edge, n, store, get_val, best_r, max_value, &counts, &sorted_vals, pl)
    }

    /// Like [`try_build_from_store`](Self::try_build_from_store), but
    /// the caller provides `r` (the bit width of the short function)
    /// directly, along with the pre-computed frequency data.
    ///
    /// This is a low-level method that requires a thorough understanding
    /// of the builder's internal state.
    ///
    /// # Arguments
    ///
    /// * `r` — bit width of the short function. The escape sentinel
    ///   is 2*ʳ* − 1 and the first 2*ʳ* − 1 entries of `sorted_vals`
    ///   become the remap table.
    /// * `max_value` — the maximum value returned by `get_val`.
    /// * `counts` — `counts[v]` is the number of keys with value `v`.
    ///   Must have length ≥ `max_value + 1`.
    /// * `sorted_vals` — distinct values sorted by descending frequency.
    pub fn try_build_from_store_with_r<V: BinSafe + Default + Send + Sync + Copy>(
        seed: u64,
        shard_edge: E,
        n: usize,
        store: &mut impl ShardStore<S, V>,
        get_val: &(impl Fn(V) -> usize + Send + Sync),
        r: usize,
        max_value: usize,
        counts: &[usize],
        sorted_vals: &[usize],
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> anyhow::Result<Self>
    where
        SigVal<S, V>: RadixKey,
        SigVal<E::LocalSig, V>: BitXor + BitXorAssign,
        SigVal<<Fuse3Shards as ShardEdge<S, 3>>::LocalSig, V>: BitXor + BitXorAssign,
    {
        let w = (max_value as u128).bit_len() as usize;
        let m = sorted_vals.len();
        let escape = (1usize << r).wrapping_sub(1); // 2^r - 1
        let num_remapped = escape.min(m);

        // -- Build remap and inv_map --

        let remap: Box<[usize]> = sorted_vals[..num_remapped].to_vec().into_boxed_slice();
        let mut inv_map = vec![usize::MAX; max_value + 1];
        for (i, &val) in remap.iter().enumerate() {
            inv_map[val] = i;
        }

        pl.info(format_args!(
            "Two-step: r={r}, escape={escape}, {num_remapped} remapped values, \
             {m} distinct values, max_value={max_value} ({w} bits)"
        ));

        // -- Build short VFunc --
        // When r = 0, escape = 0 and the short function maps every key to
        // 0 = escape, so the long function is always queried.

        pl.info(format_args!(
            "Building key -> remapped index ({r} bits, escape={escape})..."
        ));
        let short = VBuilder::<usize, BitFieldVec<Box<[usize]>>, S, E>::default()
            .try_build_func_with_store::<T, V>(
                seed,
                shard_edge,
                n,
                escape,
                store,
                &|_e, sig_val| {
                    let val = get_val(sig_val.val);
                    let mapped = inv_map[val];
                    if mapped != usize::MAX { mapped } else { escape }
                },
                pl,
            )?;

        // -- Build long VFunc (escaped keys only) --

        pl.info(format_args!(
            "Building key -> full value ({w} bits, escaped keys only)..."
        ));
        let n_escaped = n - sorted_vals[..num_remapped]
            .iter()
            .map(|&v| counts[v])
            .sum::<usize>();
        let mut long_shard_edge = Fuse3Shards::default();
        long_shard_edge.set_up_shards(n_escaped, 0.001);
        let long_shard_high_bits = long_shard_edge.shard_high_bits();

        let mut filtered_store =
            FilteredShardStore::new(store, long_shard_high_bits, |sv: &SigVal<S, V>| {
                inv_map[get_val(sv.val)] == usize::MAX
            });
        let long = VBuilder::<usize, BitFieldVec<Box<[usize]>>, S, Fuse3Shards>::default()
            .try_build_func_with_store::<T, V>(
                seed,
                long_shard_edge,
                n_escaped,
                max_value,
                &mut filtered_store,
                &|_e, sig_val| get_val(sig_val.val),
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
