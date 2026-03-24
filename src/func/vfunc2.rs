/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

#![allow(clippy::type_complexity)]

use std::borrow::Borrow;

use super::shard_edge::FuseLge3Shards;
use crate::bits::BitFieldVec;
use crate::func::VFunc;
use crate::func::shard_edge::{Fuse3NoShards, ShardEdge};
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
/// additional access to the long function.
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
    /// First function: maps each key to a remapped index (r bits), or
    /// `escape` for infrequent values. `None` when r = 0.
    pub(crate) short: Option<VFunc<T, usize, BitFieldVec<Box<[usize]>>, S, E>>,
    /// Second function: maps escaped keys to their full value.
    /// Uses Fuse3NoShards so the graph is sized for the small number of escaped
    /// keys.
    pub(crate) long: VFunc<T, usize, BitFieldVec<Box<[usize]>>, S, Fuse3NoShards>,
    /// Maps remapped indices (0..escape-1) back to actual values.
    pub(crate) remap: Box<[usize]>,
    /// The escape value (2^r - 1).
    pub(crate) escape: usize,
}

impl<T: ?Sized + ToSig<S>, S: Sig, E: ShardEdge<S, 3>> VFunc2<T, S, E>
where
    Fuse3NoShards: ShardEdge<S, 3>,
{
    /// Retrieves the value for a key given its pre-computed signature.
    ///
    /// The signature must have been computed with the same seed as the
    /// VFuncs inside (typically from a shared shard store).
    #[inline]
    pub fn get_by_sig(&self, sig: S) -> usize {
        if let Some(ref short) = self.short {
            let idx = short.get_by_sig(sig);
            if idx != self.escape {
                return self.remap[idx];
            }
        }
        self.long.get_by_sig(sig)
    }

    /// Retrieves the value associated with the given key, or an arbitrary
    /// value if the key was not in the original set.
    #[inline(always)]
    pub fn get(&self, key: impl Borrow<T>) -> usize
    where
        E: ShardEdge<S, 3>,
    {
        // Both VFuncs share the same seed; pick either.
        let seed = if let Some(ref short) = self.short {
            short.seed
        } else {
            self.long.seed
        };
        self.get_by_sig(T::to_sig(key.borrow(), seed))
    }
}

#[cfg(feature = "rayon")]
use {
    crate::func::VBuilder,
    dsi_progress_logger::ProgressLog,
    rdst::RadixKey,
    std::ops::{BitXor, BitXorAssign},
};

#[cfg(feature = "rayon")]
impl<T, S, E> VFunc2<T, S, E>
where
    T: ?Sized + ToSig<S> + std::fmt::Debug,
    S: Sig + Send + Sync,
    E: ShardEdge<S, 3>,
    Fuse3NoShards: ShardEdge<S, 3>,
    SigVal<S, usize>: RadixKey,
    SigVal<E::LocalSig, usize>: BitXor + BitXorAssign,
    SigVal<<Fuse3NoShards as ShardEdge<S, 3>>::LocalSig, usize>: BitXor + BitXorAssign,
{
    /// Builds a [`VFunc2`] from an existing shard store, applying a
    /// user-supplied `get_val` closure to extract the actual value from each
    /// store entry.
    ///
    /// # Arguments
    ///
    /// * `seed` – the seed used when hashing keys into the store.
    /// * `shard_edge` – the shard/edge configuration matching the store.
    /// * `n` – the total number of keys in the store.
    /// * `store` – a mutable reference to the shard store populated during a
    ///   prior build.
    /// * `get_val` – closure that extracts the actual value from the store's
    ///   packed value (e.g., `|v| v >> log2_bs` for LCP lengths).
    /// * `pl` – a progress logger.
    pub(crate) fn try_build_from_store<V: BinSafe + Default + Send + Sync + Copy>(
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
        SigVal<<Fuse3NoShards as ShardEdge<S, 3>>::LocalSig, V>: BitXor + BitXorAssign,
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

        // -- 5. Build short VFunc (if r > 0) --

        let short = if best_r > 0 {
            pl.info(format_args!(
                "Building key -> remapped index ({best_r} bits, escape={escape})..."
            ));
            let s = VBuilder::<usize, BitFieldVec<Box<[usize]>>, S, E>::default()
                .try_build_func_from_store::<T, V>(
                    seed,
                    shard_edge,
                    n,
                    escape, // max value is escape
                    store,
                    &|_e, sig_val| {
                        let val = get_val(sig_val.val);
                        inv_map.get(&val).copied().unwrap_or(escape)
                    },
                    pl,
                )?;
            Some(s)
        } else {
            None
        };

        // -- 6. Build long VFunc (escaped keys only) --

        pl.info(format_args!(
            "Building key -> full value ({w} bits, escaped keys only)..."
        ));
        let long = VBuilder::<usize, BitFieldVec<Box<[usize]>>, S, Fuse3NoShards>::default()
            .try_build_func_from_store_filtered::<T, V>(
                seed,
                Fuse3NoShards::default(),
                max_value,
                store,
                &|_e, sig_val| {
                    let val = get_val(sig_val.val);
                    if inv_map.contains_key(&val) {
                        None // frequent -> handled by short function
                    } else {
                        Some(val)
                    }
                },
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
