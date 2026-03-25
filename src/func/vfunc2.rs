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

impl<T: ?Sized + ToSig<S>, S: Sig, E: ShardEdge<S, 3>> VFunc2<T, S, E>
where
    Fuse3Shards: ShardEdge<S, 3>,
{
    /// Retrieves the value for a key given its pre-computed signature.
    ///
    /// The signature must have been computed with the same seed as the
    /// [`VFunc`]'s inside (typically from a shared shard store).
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
    /// Builds a [`VFunc2`] from keys and values.
    ///
    /// The keys must be provided as a rewindable lender. The values must
    /// be provided as a rewindable lender of `usize`. Internally, a
    /// temporary VFunc is built to populate the store, then the two-step
    /// analysis is applied.
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

    /// Like [`try_new`](Self::try_new), but uses the given [`VBuilder`]
    /// for the internal VFunc (e.g., offline construction or thread
    /// control).
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
        // TODO: replace _try_build_func (which solves a throwaway VFunc)
        // with a populate-only method once the retry loop is refactored.
        let (seed_vfunc, store) = builder
            .expected_num_keys(n)
            ._try_build_func::<T, B>(keys, values, true, pl)?;

        let seed = seed_vfunc.seed;
        let shard_edge = seed_vfunc.shard_edge;
        let mut store = match store {
            Some(AnyShardStore::Online(s)) => s,
            _ => unreachable!("keep_store=true"),
        };

        Self::try_build_from_store::<usize>(seed, shard_edge, n, &mut store, &|v| v, pl)
    }

    /// Builds a [`VFunc2`] from an existing shard store, applying a
    /// user-supplied `get_val` closure to extract the actual value from each
    /// store entry.
    ///
    /// # Arguments
    ///
    /// * `seed` – the seed used when hashing keys into the store.
    ///
    /// * `shard_edge` – the shard/edge configuration matching the store.
    ///
    /// * `n` – the total number of keys in the store.
    ///
    /// * `store` – a mutable reference to the shard store populated during a
    ///   prior build.
    ///
    /// * `get_val` – closure that extracts the actual value from the store's
    ///   packed value (e.g., `|v| v >> log2_bs` for LCP lengths).
    ///
    /// * `pl` – a progress logger.
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
        // -- 1. Frequency analysis --

        let mut counts: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        let mut max_value = 0usize;

        for shard in store.iter() {
            for sv in shard.iter() {
                let val = get_val(sv.val);
                *counts.entry(val).or_insert(0) += 1;
                max_value = max_value.max(val);
            }
        }

        // -- 2. Sort distinct values by descending frequency --

        let mut sorted_vals: Vec<usize> = counts.keys().copied().collect();
        sorted_vals.sort_by(|a, b| counts[b].cmp(&counts[a]));

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
                post -= counts[&sorted_vals[pos]];
                pos += 1;
            }
        }

        // Cap r to avoid huge remap arrays.
        if best_r >= usize::BITS as usize {
            best_r = usize::BITS as usize - 1;
        }

        let escape = (1usize << best_r).wrapping_sub(1); // 2^r - 1
        let num_remapped = escape; // number of values in the remap

        // -- 4. Build remap and inv_map --

        let remap: Box<[usize]> = sorted_vals[..num_remapped].to_vec().into_boxed_slice();
        let mut inv_map: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::with_capacity(num_remapped);
        for (i, &val) in remap.iter().enumerate() {
            inv_map.insert(val, i);
        }

        pl.info(format_args!(
            "Two-step: r={best_r}, escape={escape}, {num_remapped} remapped values, \
             {m} distinct values, max_value={max_value} ({w} bits)"
        ));

        // -- 5. Build short VFunc --
        // When r = 0, escape = 0 and the short function maps every key to
        // 0 = escape, so the long function is always queried.

        pl.info(format_args!(
            "Building key -> remapped index ({best_r} bits, escape={escape})..."
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
                    inv_map.get(&val).copied().unwrap_or(escape)
                },
                pl,
            )?;

        // -- 6. Build long VFunc (escaped keys only) --

        pl.info(format_args!(
            "Building key -> full value ({w} bits, escaped keys only)..."
        ));
        // Compute the optimal shard_high_bits for the escaped key count.
        let n_escaped = n - sorted_vals[..num_remapped]
            .iter()
            .map(|v| counts[v])
            .sum::<usize>();
        let mut long_shard_edge = Fuse3Shards::default();
        long_shard_edge.set_up_shards(n_escaped, 0.001);
        let long_shard_high_bits = long_shard_edge.shard_high_bits();

        let mut filtered_store =
            FilteredShardStore::new(store, long_shard_high_bits, |sv: &SigVal<S, V>| {
                !inv_map.contains_key(&get_val(sv.val))
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
