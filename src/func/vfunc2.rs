/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

#![allow(clippy::type_complexity)]

use std::borrow::Borrow;

use super::shard_edge::FuseLge3Shards;
use crate::bits::BitFieldVec;
use crate::func::VFunc;
use crate::func::shard_edge::ShardEdge;
use crate::traits::Word;
use crate::utils::*;
use mem_dbg::*;
use num_primitive::{PrimitiveNumber, PrimitiveNumberAs};
use value_traits::slices::SliceByValue;

/// A two-step static function that stores frequent values in a narrow first
/// function and infrequent values in a wider second function, with
/// frequency-based remapping.
///
/// During construction, the value distribution is analyzed and the most
/// frequent values are assigned compact indices (0 . . 2*ʳ* − 1). A first
/// [`VFunc`] maps every key either to one of these compact indices or to the
/// escape sentinel 2*ʳ* − 1. For escaped keys a second [`VFunc`], built only
/// over the escaped subset, stores the full value. A small `remap` table
/// converts compact indices back to the original values.
///
/// The split point bit width of the first function *r* is chosen to minimize
/// the total estimated space.
///
/// If the distribution of output values is very skewed, this function will use
/// much less space than a [`VFunc`]. The impact on query time is limited to the
/// additional, but rare, access to the second function; sometimes, the reduction
/// in space can even lead to faster queries due to better cache locality.
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
/// The `vfunc2` binary (`cli` feature) builds and serializes instances of
/// this structure.
///
/// # Alternatives
///
/// - A [`VFunc`] is preferable when values are not skewed, as it uses a
///   single function.
///
/// - A [`CompVFunc`] stores values in space close to their empirical
///   entropy, which is usually smaller, but queries are slower.
///
/// - [`Lcp2MmphfInt`]/[`Lcp2Mmphf`] use the same two-step technique to map
///   sorted keys to their rank.
///
/// - Contrarily to a [`VFunc`], a [`VFunc2`] cannot be wrapped in a
///   [`SignedFunc`].
///
/// # Generics
///
/// * `K` - the type of the keys; it must implement [`ToSig`] (see the
///   [`VFunc`] documentation).
///
/// * `D` - the [`SliceByValue`] storing the function data. The output value type is
///   [`D::Value`]; the default is [`BitFieldVec<Box<[usize]>>`](BitFieldVec).
///
/// * `S` - the signature type; the default is `[u64; 2]`.
///
/// * `E` - the sharding and edge logic type for the first (frequent-value)
///   function; the default is [`FuseLge3Shards`].
///
/// * `F` - the sharding and edge logic type for the second (escaped-value)
///   function; the default is `E`.
///
/// # References
///
/// Djamal Belazzougui, Paolo Boldi, Rasmus Pagh, and Sebastiano Vigna.
/// [Theory and practice of monotone minimal perfect hashing]. *ACM Journal of
/// Experimental Algorithmics*, 16(3):3.2:1−3.2:26, 2011.
///
/// # Examples
///
/// See [`try_new`].
///
/// [`try_new`]: VFunc2::try_new
/// [ε-serde]: https://crates.io/crates/epserde
/// [Theory and practice of monotone minimal perfect hashing]: https://doi.org/10.1145/1963190.2025378
/// [`serde`]: https://crates.io/crates/serde
/// [`D::Value`]: SliceByValue::Value
/// [`VBuilder`]: crate::func::VBuilder
/// [`CompVFunc`]: crate::func::CompVFunc
/// [`Lcp2MmphfInt`]: crate::func::Lcp2MmphfInt
/// [`Lcp2Mmphf`]: crate::func::Lcp2Mmphf
/// [`SignedFunc`]: crate::func::SignedFunc
#[derive(MemSize, MemDbg)]
#[cfg_attr(
    feature = "epserde",
    derive(epserde::Epserde),
    epserde(phantom(K, S)),
    epserde(bound(
        deser = "D::Value: for<'a> epserde::deser::DeserInner<DeserType<'a> = D::Value>, for<'a> <D as epserde::deser::DeserInner>::DeserType<'a>: SliceByValue<Value = D::Value>"
    ))
)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "D: serde::Serialize, D::Value: serde::Serialize, E: serde::Serialize, F: serde::Serialize",
        deserialize = "D: serde::Deserialize<'de>, D::Value: serde::Deserialize<'de>, E: serde::Deserialize<'de>, F: serde::Deserialize<'de>"
    ))
)]
pub struct VFunc2<
    K: ?Sized,
    D: SliceByValue = BitFieldVec<Box<[usize]>>,
    S = [u64; 2],
    E = FuseLge3Shards,
    F = E,
> {
    /// First function: maps each key to an index (*r* bits), or [`escape`] for
    /// infrequent values. When *r* = 0 the escape value is 0, so this is a
    /// minimal one-bit function that maps every key to 0 = escape and the
    /// second function is always queried.
    ///
    /// [`escape`]: Self::escape
    pub(crate) first: VFunc<K, D, S, E>,
    /// Second function: maps escaped keys to their full value.
    pub(crate) second: VFunc<K, D, S, F>,
    /// Maps indices [0 . . `escape`) back to actual frequent values.
    /// entries beyond the frequent prefix are zero padding.
    pub(crate) remap: Box<[D::Value]>,
    /// The escape value (2*ʳ* − 1). When *r* = 0, this value is zero and the
    /// first function always returns zero.
    pub(crate) escape: D::Value,
}

// Clone is implemented by hand, as a derived implementation would require K
// to implement Clone, too (e.g., str is not Clone)
impl<K: ?Sized, D: SliceByValue + Clone, S, E: Clone, F: Clone> Clone for VFunc2<K, D, S, E, F>
where
    D::Value: Clone,
{
    fn clone(&self) -> Self {
        Self {
            first: self.first.clone(),
            second: self.second.clone(),
            remap: self.remap.clone(),
            escape: self.escape.clone(),
        }
    }
}

impl<K: ?Sized, D: SliceByValue, S, E, F> VFunc2<K, D, S, E, F> {
    /// Returns the number of keys in the function.
    pub fn len(&self) -> usize {
        self.first.num_keys
    }

    /// Returns whether the function has no keys.
    pub fn is_empty(&self) -> bool {
        self.first.num_keys == 0
    }

    /// Returns the seed used to compute the signatures of the keys.
    ///
    /// The signatures passed to [`get_by_sig`] must be computed with this
    /// seed, that is, as `K::to_sig(key, func.seed())`.
    ///
    /// [`get_by_sig`]: VFunc2::get_by_sig
    pub const fn seed(&self) -> u64 {
        self.first.seed
    }
}

impl<K: ?Sized, D: SliceByValue, S, E, F> std::fmt::Debug for VFunc2<K, D, S, E, F>
where
    D::Value: std::fmt::Debug,
    VFunc<K, D, S, E>: std::fmt::Debug,
    VFunc<K, D, S, F>: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VFunc2")
            .field("first", &self.first)
            .field("second", &self.second)
            .field("remap", &self.remap)
            .field("escape", &self.escape)
            .finish()
    }
}

impl<K: ?Sized, W: Word, S: Sig, E: ShardEdge<S, 3>, F: ShardEdge<S, 3>>
    VFunc2<K, BitFieldVec<Box<[W]>>, S, E, F>
{
    /// Creates a VFunc2 with zero keys.
    ///
    /// With `escape = 0`, the first function always returns the escape,
    /// so `get` always queries the second function (which returns zero
    /// since both internal [`VFunc`]s are empty).
    #[must_use]
    pub fn empty() -> Self {
        Self {
            first: VFunc::empty(),
            second: VFunc::empty(),
            remap: Box::new([]),
            escape: W::ZERO,
        }
    }
}

impl<
    K: ?Sized + ToSig<S>,
    D: SliceByValue<Value: Word + BinSafe + PrimitiveNumberAs<usize>>,
    S: Sig,
    E: ShardEdge<S, 3>,
    F: ShardEdge<S, 3>,
> VFunc2<K, D, S, E, F>
{
    /// Retrieves the value for a key given its pre-computed signature.
    ///
    /// The signature must have been computed with the [seed](VFunc2::seed)
    /// of the function, that is, as `K::to_sig(key, func.seed())`.
    ///
    /// This method is mainly useful in the construction of compound
    /// functions.
    #[inline]
    pub fn get_by_sig(&self, sig: S) -> D::Value {
        let idx = self.first.get_by_sig(sig);
        // remap has exactly escape entries, so this is equivalent to
        // comparing idx with the escape sentinel, at the same cost; it
        // additionally covers the degenerate r = 0 regime, in which the
        // one-bit-wide first function could in principle return an index
        // beyond the (empty) remapping array.
        match self.remap.get(idx.as_to::<usize>()) {
            Some(&value) => value,
            None => self.second.get_by_sig(sig),
        }
    }

    /// Retrieves the value associated with the given key, or an arbitrary
    /// value if the key was not in the original set.
    #[inline(always)]
    pub fn get(&self, key: impl Borrow<K>) -> D::Value {
        self.get_by_sig(K::to_sig(key.borrow(), self.first.seed))
    }
}

// ── Aligned ↔ Unaligned conversions ─────────────────────────────────

use crate::traits::{TryIntoUnaligned, Unaligned};

impl<K: ?Sized, W: Word, S: Sig, E: ShardEdge<S, 3>, F: ShardEdge<S, 3>> TryIntoUnaligned
    for VFunc2<K, BitFieldVec<Box<[W]>>, S, E, F>
{
    type Unaligned = VFunc2<K, Unaligned<BitFieldVec<Box<[W]>>>, S, E, F>;
    fn try_into_unaligned(
        self,
    ) -> Result<Self::Unaligned, crate::traits::UnalignedConversionError> {
        Ok(VFunc2 {
            first: self.first.try_into_unaligned()?,
            second: self.second.try_into_unaligned()?,
            remap: self.remap,
            escape: self.escape,
        })
    }
}

impl<K: ?Sized, W: Word, S: Sig, E: ShardEdge<S, 3>, F: ShardEdge<S, 3>>
    From<Unaligned<VFunc2<K, BitFieldVec<Box<[W]>>, S, E, F>>>
    for VFunc2<K, BitFieldVec<Box<[W]>>, S, E, F>
{
    fn from(vf: Unaligned<VFunc2<K, BitFieldVec<Box<[W]>>, S, E, F>>) -> Self {
        VFunc2 {
            first: VFunc::from(vf.first),
            second: VFunc::from(vf.second),
            remap: vf.remap,
            escape: vf.escape,
        }
    }
}

#[cfg(feature = "rayon")]
mod build {
    use super::*;
    use crate::func::{VBuilder, check_num_values};
    use core::error::Error;
    use dsi_progress_logger::ProgressLog;
    use lender::*;
    use rdst::RadixKey;
    use std::ops::{BitXor, BitXorAssign};

    /// A map from `K` to `V` backed by a flat array for small keys and a
    /// [`HashMap`] for large keys.
    pub(crate) struct HybridMap<K, V> {
        array: Vec<V>,
        map: std::collections::HashMap<K, V>,
        default: V,
    }

    impl<K: Word + PrimitiveNumberAs<usize> + PrimitiveNumberAs<u128>, V: Copy + Eq> HybridMap<K, V> {
        /// Returns the flat-array index for `key`, or `None` when `key` does
        /// not fit in a `usize`. Guarding on the full-width value prevents two
        /// distinct wide keys (e.g. `u128` values differing only in their high
        /// bits) from truncating to the same array slot.
        #[inline(always)]
        fn key_index(key: K) -> Option<usize> {
            usize::try_from(key.as_to::<u128>()).ok()
        }

        /// Creates a new hybrid map.
        ///
        /// * `max_key` - optional upper bound on keys. When provided,
        ///   the array is capped at `max_key + 1`.
        /// * `default` - value returned for absent keys.
        pub(crate) fn new(max_key: Option<K>, default: V) -> Self {
            let mut array_len = 1 << 10;
            if let Some(mk) = max_key {
                if let Some(k) = Self::key_index(mk) {
                    array_len = array_len.min(k.saturating_add(1));
                }
            }
            Self {
                array: vec![default; array_len],
                map: std::collections::HashMap::new(),
                default,
            }
        }

        pub(crate) fn insert(&mut self, key: K, value: V) {
            match Self::key_index(key) {
                Some(k) if k < self.array.len() => self.array[k] = value,
                _ => {
                    self.map.insert(key, value);
                }
            }
        }

        #[inline(always)]
        pub(crate) fn get(&self, key: K) -> V {
            match Self::key_index(key) {
                Some(k) if k < self.array.len() => self.array[k],
                _ => self.map.get(&key).copied().unwrap_or(self.default),
            }
        }

        /// Returns keys whose value differs from the default, sorted by
        /// descending value (requires `V: Ord`).
        pub(crate) fn keys_by_desc_value(&self) -> Vec<K>
        where
            V: Ord,
        {
            let array_iter = self
                .array
                .iter()
                .enumerate()
                .filter(|&(_, v)| *v != self.default)
                .map(|(k, _)| K::try_from(k).ok().unwrap());
            let map_iter = self.map.keys().copied();
            let mut keys: Vec<K> = array_iter.chain(map_iter).collect();
            keys.sort_by_key(|b| std::cmp::Reverse(self.get(*b)));
            keys
        }
    }

    impl<K: Word + PrimitiveNumberAs<usize> + PrimitiveNumberAs<u128>> HybridMap<K, usize> {
        #[inline(always)]
        pub(crate) fn incr(&mut self, key: K) {
            self.add(key, 1);
        }

        #[inline(always)]
        pub(crate) fn add(&mut self, key: K, amount: usize) {
            match Self::key_index(key) {
                Some(k) if k < self.array.len() => self.array[k] += amount,
                _ => *self.map.entry(key).or_insert(0) += amount,
            }
        }
    }

    /// Finds the optimal first-function bit width `r` for a [`VFunc2`],
    /// minimizing the estimated total space.
    ///
    /// `sorted_vals` must be the distinct values sorted by descending
    /// frequency. `count_of` returns the frequency for a given value.
    pub(crate) fn find_optimal_r<W: Word>(
        n: usize,
        max_value: W,
        sorted_vals: &[W],
        count_of: impl Fn(W) -> usize,
        w_bits: usize,
    ) -> usize {
        let w = max_value.bit_len() as usize;
        let m = sorted_vals.len();
        let c = 1.11f64; // VFunc expansion factor (approximate)

        let mut post = n;
        let mut pos = 0usize;
        let mut best_r = 0usize;
        let mut best_cost = f64::MAX;

        // r <= w <= 128. r == w is the pure-first-function configuration, which
        // is optimal when the alphabet fills most of the w-bit space (so no
        // smaller width absorbs every distinct value). checked_shl keeps r == 128
        // (a full u128 value width) from overflowing.
        for r in 0..=w {
            // The first function is built with escape = (1 << r) - 1 as its
            // max value, so its bit width is bit_len(escape): r for r >= 1 and
            // 1 for r == 0 (bit_len(0) == 1). Charge the real one-bit cost at
            // r == 0 so the optimizer does not treat the still-built one-bit
            // first stage as free and over-select r == 0.
            let first_width = if r == 0 { 1 } else { r };
            let cost_first = c * n as f64 * first_width as f64;
            let cost_second = c * post as f64 * w as f64;
            // The remap table is allocated with escape = 2^r - 1 entries
            // (frequent values plus zero padding for first-function outputs in
            // [m, escape)), so charge its full length, not just the used prefix.
            let escape = 2f64.powi(i32::try_from(r).expect("r fits i32")) - 1.0;
            let cost_remap = escape * w_bits as f64;
            let cost = cost_first + cost_second + cost_remap;

            if cost < best_cost {
                best_cost = cost;
                best_r = r;
            }

            let to_absorb = 1u128
                .checked_shl(u32::try_from(r).expect("r fits u32"))
                .and_then(|v| usize::try_from(v).ok())
                .unwrap_or(usize::MAX)
                .min(m - pos);
            for _ in 0..to_absorb {
                post -= count_of(sorted_vals[pos]);
                pos += 1;
            }
        }

        best_r
    }

    /// The frequency-based remapping of the values of a two-step function.
    ///
    /// Frequent values are remapped to indices in [0 . . `escape`), and the
    /// remaining values to `escape`.
    pub(crate) struct Remap<W> {
        /// The escape value 2*ʳ* − 1, where *r* is chosen by
        /// [`find_optimal_r`].
        pub(crate) escape: W,
        /// Maps indices in [0 . . `escape`) back to frequent values; entries
        /// beyond the frequent values are zero padding.
        pub(crate) remap: Box<[W]>,
        /// Maps frequent values to their indices, and other values to
        /// `escape`.
        pub(crate) inv_map: HybridMap<W, W>,
        /// The number of keys whose value is not frequent.
        pub(crate) num_escaped: usize,
    }

    impl<W: Word + PrimitiveNumberAs<usize> + PrimitiveNumberAs<u128>> Remap<W> {
        /// Chooses the escape minimizing the estimated total space, given
        /// the number of occurrences of each value, and builds the
        /// remapping.
        pub(crate) fn new(
            counts: &HybridMap<W, usize>,
            num_keys: usize,
            max_value: W,
            pl: &mut impl ProgressLog,
        ) -> Self {
            let sorted_vals: Vec<W> = counts.keys_by_desc_value();
            let num_distinct = sorted_vals.len();

            let r = find_optimal_r(
                num_keys,
                max_value,
                &sorted_vals,
                |v| counts.get(v),
                W::BITS as usize,
            );

            // Shift in u128 so r >= 32 cannot overflow a 32-bit usize.
            let escape_usize = usize::try_from(
                1u128
                    .checked_shl(r as u32)
                    .map(|v| v - 1)
                    .unwrap_or(u128::MAX),
            )
            .expect("escape range exceeds usize");
            let escape = W::try_from(escape_usize).ok().unwrap();
            let num_frequent = escape_usize.min(num_distinct);

            // We must cover the possible outputs of the first function in
            // [num_frequent . . escape), including the arbitrary outputs of
            // absent keys, so the tail of remap stays zeroed as padding.
            let mut remap: Box<[W]> = vec![W::ZERO; escape_usize].into();
            remap[..num_frequent].copy_from_slice(&sorted_vals[..num_frequent]);
            // Build inv_map from the frequent prefix only, so the zero padding
            // does not pollute it (missing values resolve to the escape sentinel).
            let mut inv_map: HybridMap<W, W> = HybridMap::new(Some(max_value), escape);
            for (i, &val) in remap[..num_frequent].iter().enumerate() {
                inv_map.insert(val, W::try_from(i).ok().unwrap());
            }

            let num_escaped = num_keys
                - sorted_vals[..num_frequent]
                    .iter()
                    .map(|&v| counts.get(v))
                    .sum::<usize>();

            pl.info(format_args!(
                "r: {r}; distinct values: {num_distinct}; frequent values: {num_frequent} ({:.3}%); max value: {max_value} ({} bits)",
                if num_distinct == 0 {
                    0.0
                } else {
                    100.0 * num_frequent as f64 / num_distinct as f64
                },
                max_value.bit_len()
            ));

            Self {
                escape,
                remap,
                inv_map,
                num_escaped,
            }
        }
    }

    impl<
        K: ?Sized + ToSig<S> + std::fmt::Debug,
        W: Word + BinSafe + MemSize + mem_dbg::FlatType,
        S: Sig + Send + Sync,
        E: ShardEdge<S, 3> + MemSize + mem_dbg::FlatType,
        F: ShardEdge<S, 3> + MemSize + mem_dbg::FlatType,
    > VFunc2<K, BitFieldVec<Box<[W]>>, S, E, F>
    where
        Box<[W]>: MemSize,
        SigVal<S, W>: RadixKey,
        SigVal<E::LocalSig, W>: BitXor + BitXorAssign,
        SigVal<F::LocalSig, W>: BitXor + BitXorAssign,
    {
        /// Builds a [`VFunc2`] from keys and values using default [`VBuilder`]
        /// settings.
        ///
        /// This is a convenience wrapper around [`try_new_with_builder`] with
        /// `VBuilder::default()`.
        ///
        /// * `keys` and `values` -
        ///   [`FallibleRewindableLender`]s, aligned (one value per key,
        ///   same order). The values lender may return more values than
        ///   there are keys (in particular, it may be infinite); the
        ///   extra values are ignored. If it returns fewer values, a
        ///   [`MismatchedKeysAndValues`] error is returned.
        ///   The [`lenders`] module provides easy ways to build such
        ///   lenders.
        ///
        /// [`MismatchedKeysAndValues`]: crate::func::BuildError::MismatchedKeysAndValues
        ///
        /// # Examples
        ///
        /// ```rust
        /// # #[cfg(feature = "rayon")]
        /// # fn main() -> anyhow::Result<()> {
        /// # use sux::func::VFunc2;
        /// # use sux::bits::BitFieldVec;
        /// # use dsi_progress_logger::no_logging;
        /// # use sux::utils::FromCloneableIntoIterator;
        /// let func: VFunc2<usize, BitFieldVec<Box<[usize]>>> = VFunc2::try_new(
        ///     FromCloneableIntoIterator::new(0..100_usize),
        ///     FromCloneableIntoIterator::new(0..100_usize),
        ///     no_logging![],
        /// )?;
        ///
        /// for i in 0..100 {
        ///     assert_eq!(func.get(&i), i);
        /// }
        /// # Ok(())
        /// # }
        /// # #[cfg(not(feature = "rayon"))]
        /// # fn main() {}
        /// ```
        ///
        /// [`try_new_with_builder`]: Self::try_new_with_builder
        pub fn try_new<B: ?Sized + std::borrow::Borrow<K>>(
            keys: impl FallibleRewindableLender<
                RewindError: Error + Send + Sync + 'static,
                Error: Error + Send + Sync + 'static,
            > + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
            values: impl FallibleRewindableLender<
                RewindError: Error + Send + Sync + 'static,
                Error: Error + Send + Sync + 'static,
            > + for<'lend> FallibleLending<'lend, Lend = &'lend W>,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> anyhow::Result<Self> {
            Self::try_new_with_builder(keys, values, VBuilder::default(), pl)
        }

        /// Builds a [`VFunc2`] from keys and values using the given
        /// [`VBuilder`] configuration.
        ///
        /// The builder controls construction parameters such as [offline
        /// mode], [thread count], [sharding overhead], and [PRNG seed].
        ///
        /// * `keys` and `values` -
        ///   [`FallibleRewindableLender`]s, aligned (one value per key,
        ///   same order). The values lender may return more values than
        ///   there are keys (in particular, it may be infinite); the
        ///   extra values are ignored. If it returns fewer values, a
        ///   [`MismatchedKeysAndValues`] error is returned.
        ///   The [`lenders`] module provides easy ways to build such
        ///   lenders.
        ///
        /// [`MismatchedKeysAndValues`]: crate::func::BuildError::MismatchedKeysAndValues
        ///
        /// # Examples
        ///
        /// ```rust
        /// # #[cfg(feature = "rayon")]
        /// # fn main() -> anyhow::Result<()> {
        /// # use sux::func::{VBuilder, VFunc2};
        /// # use sux::bits::BitFieldVec;
        /// # use dsi_progress_logger::no_logging;
        /// # use sux::utils::FromCloneableIntoIterator;
        /// let func: VFunc2<usize, BitFieldVec<Box<[usize]>>> = VFunc2::try_new_with_builder(
        ///     FromCloneableIntoIterator::new(0..100_usize),
        ///     FromCloneableIntoIterator::new(0..100_usize),
        ///     VBuilder::default().offline(true),
        ///     no_logging![],
        /// )?;
        ///
        /// for i in 0..100 {
        ///     assert_eq!(func.get(&i), i);
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
        pub fn try_new_with_builder<B: ?Sized + std::borrow::Borrow<K>>(
            keys: impl FallibleRewindableLender<
                RewindError: Error + Send + Sync + 'static,
                Error: Error + Send + Sync + 'static,
            > + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
            values: impl FallibleRewindableLender<
                RewindError: Error + Send + Sync + 'static,
                Error: Error + Send + Sync + 'static,
            > + for<'lend> FallibleLending<'lend, Lend = &'lend W>,
            builder: VBuilder<BitFieldVec<Box<[W]>>, S, E>,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> anyhow::Result<Self> {
            let mut builder = builder;
            builder
                .try_populate_and_build(
                    keys,
                    values,
                    &mut |builder, seed, mut store, _max_value, _num_keys, pl, _state: &mut ()| {
                        Self::try_build_from_store::<W>(
                            seed,
                            builder.shard_edge,
                            &mut *store,
                            &|v| v,
                            VBuilder::default().set_from(builder),
                            pl,
                        )
                    },
                    pl,
                    (),
                )
                .map(|(func, _keys)| func)
        }

        /// Builds a [`VFunc2`] from in-memory key and value slices,
        /// parallelizing hash computation and sig-store population with
        /// rayon, using default [`VBuilder`] settings.
        ///
        /// This is the parallel counterpart of [`try_new`]: each key is
        /// hashed on a rayon worker thread and deposited directly into
        /// its sig-store bucket. Faster than the lender-based path for
        /// large in-memory key sets, but requires the key and value
        /// inputs to be addressable as slices.
        ///
        /// This is a convenience wrapper around
        /// [`try_par_new_with_builder`] with `VBuilder::default()`.
        ///
        /// [`try_new`]: Self::try_new
        /// [`try_par_new_with_builder`]: Self::try_par_new_with_builder
        pub fn try_par_new<B: Borrow<K> + Sync>(
            keys: &[B],
            values: &[W],
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> anyhow::Result<Self>
        where
            K: Sync,
        {
            Self::try_par_new_with_builder(keys, values, VBuilder::default(), pl)
        }

        /// Builds a [`VFunc2`] from in-memory key and value slices,
        /// parallelizing hash computation and sig-store population with
        /// rayon, using the given [`VBuilder`] configuration.
        ///
        /// See [`try_par_new`](Self::try_par_new) for parallel-path
        /// semantics and [`try_new_with_builder`] for the lender-based
        /// variant.
        ///
        /// [`try_new_with_builder`]: Self::try_new_with_builder
        pub fn try_par_new_with_builder<B: Borrow<K> + Sync>(
            keys: &[B],
            values: &[W],
            builder: VBuilder<BitFieldVec<Box<[W]>>, S, E>,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> anyhow::Result<Self>
        where
            K: Sync,
        {
            check_num_values(keys.len(), values.len())?;
            builder
                .expected_num_keys(keys.len())
                .try_par_populate_and_build(
                    keys,
                    &|i| values[i],
                    &mut |builder, seed, mut store, _max_value, _num_keys, pl, _state: &mut ()| {
                        Self::try_build_from_store::<W>(
                            seed,
                            builder.shard_edge,
                            &mut *store,
                            &|v| v,
                            VBuilder::default().set_from(builder),
                            pl,
                        )
                    },
                    pl,
                    (),
                )
        }

        /// Builds a [`VFunc2`] from a populated [`ShardStore`].
        ///
        /// This is a low-level constructor that reuses the signatures stored
        /// by a previous construction, without hashing the keys again. Use
        /// [`try_new`] or [`try_new_with_builder`] for the common case of
        /// building from keys and values directly.
        ///
        /// # Preconditions
        ///
        /// * `seed` and `shard_edge` must be the values used when the store
        ///   was populated.
        ///
        /// * `get_val` must be deterministic: the store is iterated multiple
        ///   times (frequency analysis, then construction of the two functions) and
        ///   differing results corrupt the function silently.
        ///
        /// # Arguments
        ///
        /// * `seed` - the seed from the store's population step.
        ///
        /// * `shard_edge` - the shard edge from the same population step.
        ///
        /// * `store` - the populated shard store.
        ///
        /// * `get_val` - extracts the value from the store's packed entry.
        ///
        /// * `builder` - the builder configuration for the internal VFuncs.
        ///
        /// * `pl` - a progress logger.
        ///
        /// [`try_new`]: Self::try_new
        /// [`try_new_with_builder`]: Self::try_new_with_builder
        pub fn try_build_from_store<V: BinSafe + Default + Send + Sync + Copy>(
            seed: u64,
            shard_edge: E,
            store: &mut (impl ShardStore<S, V> + ?Sized),
            get_val: &(impl Fn(V) -> W + Send + Sync),
            mut builder: VBuilder<BitFieldVec<Box<[W]>>, S, E>,
            pl: &mut (impl ProgressLog + Clone + Send + Sync),
        ) -> anyhow::Result<Self>
        where
            SigVal<S, V>: RadixKey,
            SigVal<E::LocalSig, V>: BitXor + BitXorAssign,
            SigVal<F::LocalSig, V>: BitXor + BitXorAssign,
        {
            // -- Frequency analysis (single pass) --

            let mut max_value = W::ZERO;
            let mut counts: HybridMap<W, usize> = HybridMap::new(None, 0);
            for shard in store.iter() {
                for sv in shard.iter() {
                    let val = get_val(sv.val);
                    if val > max_value {
                        max_value = val;
                    }
                    counts.incr(val);
                }
            }

            let num_keys = store.len();
            let remap = Remap::new(&counts, num_keys, max_value, pl);
            let escaped = |v: V| remap.inv_map.get(get_val(v)) == remap.escape;

            // -- Build the first function --
            // When r = 0, escape = 0 and the first function maps every key to
            // 0 = escape, so the second function is always queried.

            pl.push_log_target(" ▸ first");
            let result = builder.try_build_first_func::<K, V>(
                seed,
                shard_edge,
                remap.escape,
                store,
                &|v| remap.inv_map.get(get_val(v)),
                &escaped,
                pl,
            );
            // Pop before propagating errors, or retries would nest log targets
            pl.pop_log_target();
            let (first, escaped_counts) = result?;

            debug_assert_eq!(
                escaped_counts.iter().sum::<usize>(),
                remap.num_escaped,
                "inspect-counted escaped != freq-computed escaped"
            );

            // -- Build the second function (escaped keys only) --

            pl.push_log_target(" ▸ second");
            let result = builder.try_build_escaped_func::<K, V, F>(
                seed,
                store,
                &escaped_counts,
                &escaped,
                max_value,
                get_val,
                pl,
            );
            pl.pop_log_target();
            let second = result?;

            let bit_width = second.data.bit_width() as f64;

            Ok(Self {
                first,
                second,
                remap: remap.remap,
                escape: remap.escape,
            })
            .inspect(|vfunc2| {
                let size = vfunc2.mem_size(SizeFlags::default()) as f64 * 8.0;
                pl.info(format_args!(
                    "Bits/key: {:.3} ({:+.3}% with respect to bit width)",
                    size / num_keys as f64,
                    100.0 * (size / (num_keys as f64 * bit_width) - 1.),
                ));
            })
        }
    }
}

#[cfg(feature = "rayon")]
pub(crate) use build::{HybridMap, Remap};

#[cfg(all(test, feature = "rayon"))]
mod tests {
    use super::build::find_optimal_r;
    use super::*;
    use crate::bits::BitFieldVec;
    use crate::utils::FromCloneableIntoIterator;
    use dsi_progress_logger::no_logging;

    /// When `find_optimal_r` picks an escape range larger than the distinct-value
    /// count (`escape > m`), the `remap` table is padded to `escape`: the frequent
    /// values fill the prefix and the tail stays zeroed. A near-uniform
    /// distribution over a few *large* distinct values forces this regime: the
    /// large value width makes escaping costly, so all `m` values are absorbed
    /// with `escape = 2^ceil(log2 m) - 1 > m` (here `m = 5`, `escape = 7`).
    ///
    /// The build must not panic (the padding must be copied into the prefix, not
    /// the whole slice), present keys must map to their correct value (the zero
    /// padding must not pollute `inv_map`), and absent keys must not panic.
    #[test]
    fn test_remap_escape_gt_m() {
        const DISTINCT: [usize; 5] = [0, 100_000, 200_000, 300_000, 400_000];
        let n = 50_000usize;
        let func: VFunc2<usize, BitFieldVec<Box<[usize]>>> = VFunc2::try_new(
            FromCloneableIntoIterator::new(0..n),
            FromCloneableIntoIterator::new((0..n).map(|i| DISTINCT[i % DISTINCT.len()])),
            no_logging![],
        )
        .unwrap();

        for i in 0..n {
            assert_eq!(
                func.get(&i),
                DISTINCT[i % DISTINCT.len()],
                "present key {i}"
            );
        }
        // Absent keys yield an arbitrary value but must never panic (the first
        // function may return an index in [num_frequent, escape) into remap).
        for i in n..n + 5_000 {
            std::hint::black_box(func.get(&i));
        }
    }

    /// Two distinct keys that share their low `usize` bits but differ in high
    /// bits (only representable in `u128`) must map to distinct entries rather
    /// than colliding on the same flat-array slot.
    #[test]
    fn hybrid_map_distinguishes_wide_keys() {
        let mut m: HybridMap<u128, u32> = HybridMap::new(None, 0);
        let low = 5u128;
        let high = (1u128 << 64) | 5; // same low 64 bits as low
        m.insert(low, 111);
        m.insert(high, 222);
        assert_eq!(m.get(low), 111);
        assert_eq!(m.get(high), 222);
        assert_eq!(m.get(7), 0); // absent -> default
    }

    /// The frequency-counter path (`add`/`incr`) must also keep wide keys
    /// distinct instead of merging their counts on a shared array slot.
    #[test]
    fn hybrid_map_add_distinguishes_wide_keys() {
        let mut m: HybridMap<u128, usize> = HybridMap::new(None, 0);
        let low = 5u128;
        let high = (1u128 << 64) | 5;
        m.add(low, 3);
        m.add(high, 7);
        m.incr(low);
        assert_eq!(m.get(low), 4);
        assert_eq!(m.get(high), 7);
    }

    /// A value width above 32 bits must not overflow the `1 << r` shift on
    /// 32-bit targets (it is computed in `u128`).
    #[test]
    fn find_optimal_r_handles_wide_values() {
        let sorted_vals: Vec<u64> = vec![1 << 40, 1 << 39, 1 << 38, 1 << 37];
        let r = find_optimal_r(400, 1u64 << 40, &sorted_vals, |_| 100, 64);
        assert!(r <= 64);
    }

    /// The first function built for `r == 0` is a one-bit function
    /// (`bit_len(0) == 1`), not a free/empty stage, so `find_optimal_r` must
    /// charge its `c * n` bits. Otherwise the old `cost_first = 0.0` model
    /// under-prices `r == 0` by `c * n` and over-selects it: here the honest
    /// cost makes `r == 1` win (absorbing the frequent value shrinks the
    /// second stage by more than the remap entry costs).
    #[test]
    fn find_optimal_r_charges_one_bit_first_stage_at_zero() {
        let sorted_vals: Vec<u64> = vec![1, 2];
        let count_of = |v: u64| match v {
            1 => 60,
            2 => 40,
            _ => 0,
        };
        let r = find_optimal_r(100, 2u64, &sorted_vals, count_of, 64);
        assert_eq!(r, 1);
    }

    /// When the alphabet fills the `w`-bit non-escape space, no width below `w`
    /// absorbs every distinct value, so `r == w` (the pure-first-function
    /// configuration with an empty second function) is optimal. Here `w == 3`
    /// (max_value 7) with all 7 non-escape ranks equally frequent: width 2
    /// absorbs only 3 values and pays for a second function over the rest, while
    /// width 3 absorbs all 7 with no second function. The old `0..w` loop could
    /// not select `r == w`, and (before the remap cost was charged for its full
    /// padded length) the model also mispriced this regime.
    #[test]
    fn find_optimal_r_selects_full_width_for_filled_alphabet() {
        let sorted_vals: Vec<u8> = vec![7, 6, 5, 4, 3, 2, 1];
        let r = find_optimal_r(
            105,
            7u8,
            &sorted_vals,
            |_| 15,
            usize::try_from(u8::BITS).unwrap(),
        );
        assert_eq!(r, 3);
    }

    /// Regression for the padded-remap cost: only 5 of the 7 non-escape ranks
    /// are present, so width 3 would allocate a 7-entry remap with two unused
    /// (padded) slots, not worth the extra first-function bit; width 2 wins.
    /// Pricing the remap by its used prefix instead of its allocated `2^r - 1`
    /// length wrongly selects width 3 here.
    #[test]
    fn find_optimal_r_prices_padded_remap() {
        let sorted_vals: Vec<u8> = vec![7, 6, 5, 4, 3];
        let r = find_optimal_r(
            100,
            7u8,
            &sorted_vals,
            |_| 20,
            usize::try_from(u8::BITS).unwrap(),
        );
        assert_eq!(r, 2);
    }
}
