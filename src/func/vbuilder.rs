/*
 * SPDX-FileCopyrightText: 2025 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]

use crate::bits::*;
use crate::func::{shard_edge::ShardEdge, *};
use crate::traits::bit_field_slice::{BitFieldSlice, BitFieldSliceMut};
use crate::traits::{BitVecOpsMut, Word};
use crate::utils::*;
use core::error::Error;
use derivative::Derivative;
use derive_setters::*;
use dsi_progress_logger::*;
use lender::FallibleLending;
use mem_dbg::{FlatType, MemSize, SizeFlags};
use num_primitive::PrimitiveNumber;
use rand::rngs::SmallRng;
use rand::{Rng, RngExt, SeedableRng};
use rayon::iter::ParallelIterator;
use rayon::slice::ParallelSlice;
use rdst::*;
use std::any::TypeId;
use std::borrow::{Borrow, Cow};
use std::marker::PhantomData;
use std::mem::transmute;
use std::ops::{BitXor, BitXorAssign};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;
use thread_priority::ThreadPriority;
use value_traits::slices::{SliceByValue, SliceByValueMut};

use super::shard_edge::FuseLge3Shards;

const LOG2_MAX_SHARDS: u32 = 16;

/// Returns the default maximum number of threads: `min(16, available_parallelism)`.
fn default_max_num_threads() -> usize {
    std::thread::available_parallelism()
        .map(|p| p.get().min(16))
        .unwrap_or(1)
}

/// A builder for [`VFunc`] and [`VFilter`].
///
/// There are two construction modes: in core memory (default) and
/// [offline]; both use a [`SigStore`]. In the first case, space will be
/// allocated in core memory for signatures and associated values for all
/// keys; in the second case, such information will be stored in a number
/// of on-disk buckets.
///
/// There are several setters: for example, you can [set the maximum number
/// of threads].
///
/// [offline]: VBuilder::offline
/// [set the maximum number of threads]: VBuilder::max_num_threads
///
/// # Implementation details
///
/// Initially, keys are scanned, turned into signatures, and stored in the
/// [`SigStore`], possibly together with associated values.
///
/// Once signatures have been computed, each parallel thread will process a
/// shard of the signature/value pairs. For very large key sets shards will be
/// significantly smaller than the number of keys, so the memory usage, in
/// particular in offline mode, can be significantly reduced. Note that using
/// too many threads might actually be harmful due to memory contention.
///
/// The generic parameters are explained in the [`VFunc`] /
/// [`VFilter`] documentation.
//
/// Most methods require to pass one or two [`FallibleRewindableLender`]s
/// (keys and possibly values), as the construction might fail and keys might
/// be scanned again. The structures in the [`lenders`] module provide easy ways
/// to build such lenders.
///
/// [`VFilter`]: crate::dict::VFilter
#[derive(Setters, Debug, Derivative)]
#[derivative(Default)]
#[setters(generate = false)]
pub struct VBuilder<D, S = [u64; 2], E = FuseLge3Shards> {
    /// The expected number of keys.
    ///
    /// While this setter is optional, setting this value to a reasonable bound
    /// on the actual number of keys will significantly speed up the
    /// construction.
    #[setters(generate = true, strip_option)]
    #[derivative(Default(value = "None"))]
    expected_num_keys: Option<usize>,

    /// Override the value used to size the sharding step.
    ///
    /// Normally [`set_up_shards`](ShardEdge::set_up_shards) is called
    /// with the number of keys, which is correct for a construction
    /// where each key produces exactly one hyperedge (VFunc). When
    /// each key produces *multiple* hyperedges — as in
    /// [`CompVFunc`](crate::func::CompVFunc), where each key's
    /// Huffman-encoded value contributes one edge per codeword bit
    /// — the graph actually has ≈ entropy × num_keys equations.
    /// Sizing the shards for the smaller key count leaves every
    /// shard oversized relative to the sharding strategy's intent.
    ///
    /// If this field is `Some(n)`, `n` is passed to
    /// [`set_up_shards`](ShardEdge::set_up_shards) instead of the
    /// runtime key count. CompVFunc sets this to the total edge
    /// count after building the Huffman coder.
    #[setters(generate = true, strip_option)]
    #[derivative(Default(value = "None"))]
    pub(crate) shard_size_hint: Option<usize>,

    /// The maximum number of parallel threads to use for both the population
    /// and solve phases. The default is `min(16, available_parallelism)`.
    #[setters(generate = true)]
    #[derivative(Default(value = "default_max_num_threads()"))]
    pub(crate) max_num_threads: usize,

    /// Use disk-based buckets to reduce core memory usage at construction time.
    #[setters(generate = true)]
    offline: bool,

    /// Check for duplicated signatures. This is not necessary in general,
    /// but if you suspect you might be feeding duplicate keys, you can
    /// enable this check.
    #[setters(generate = true)]
    check_dups: bool,

    /// Use always the low-memory peel-by-signature algorithm (true) or the
    /// high-memory peel-by-index algorithm (false). The former is slightly
    /// slower, but it uses much less memory. Normally [`VBuilder`] uses
    /// high-mem and switches to low-mem if there are more
    /// than three threads and more than two shards.
    #[setters(generate = true, strip_option)]
    #[derivative(Default(value = "None"))]
    pub(crate) low_mem: Option<bool>,

    /// The seed for the random number generator.
    #[setters(generate = true)]
    seed: u64,

    /// The base-2 logarithm of buckets of the [`SigStore`]. The default is 8.
    /// This value is automatically overridden, even if set, if you provide an
    /// [expected number of keys].
    ///
    /// [expected number of keys]: VBuilder::expected_num_keys
    #[setters(generate = true, strip_option)]
    #[derivative(Default(value = "8"))]
    log2_buckets: u32,

    /// The target relative space loss due to [ε-cost sharding].
    ///
    /// The default is 0.001. Setting a larger target, for example, 0.01, will
    /// increase the space overhead due to sharding, but will provide in general
    /// finer shards. This might not always happen, however, because the ε-cost
    /// bound is just one of the bounds concurring in limiting the amount of
    /// sharding for a specific [`ShardEdge`]. For example, increasing the
    /// target to 0.01 will provide very fine sharding using `Mwhc3Shards`
    /// shard/edge logic, activated by the `mwhc` feature, but will affect only
    /// slightly [`FuseLge3Shards`] or [`FuseLge3FullSigs`].
    ///
    /// [`FuseLge3FullSigs`]: crate::func::shard_edge::FuseLge3FullSigs
    ///
    /// [ε-cost sharding]: https://doi.org/10.4230/LIPIcs.ESA.2019.38
    #[setters(generate = true, strip_option)]
    #[derivative(Default(value = "0.001"))]
    pub(crate) eps: f64,

    /// The bit width of the maximum value.
    pub(crate) bit_width: usize,
    /// The edge generator.
    pub(crate) shard_edge: E,
    /// The number of keys.
    pub(crate) num_keys: usize,
    /// The ratio between the number of vertices and the number of edges
    /// (i.e., between the number of variables and the number of equations).
    pub(crate) c: f64,
    /// Whether we should use [lazy Gaussian elimination].
    ///
    /// [lazy Gaussian elimination]: https://doi.org/10.1016/j.ic.2020.104517
    pub(crate) lge: bool,
    /// The number of threads to use.
    pub(crate) num_threads: usize,
    /// Fast-stop for failed attempts.
    pub(crate) failed: AtomicBool,
    #[doc(hidden)]
    _marker: PhantomData<(D, S)>,
}

impl<D: BitFieldSlice<Value: Word + BinSafe> + Send + Sync, S: Sig, E: ShardEdge<S, 3>>
    VBuilder<D, S, E>
{
    /// Sets up shards from the expected number of keys and returns the
    /// seed. This is the same initialization that
    /// [`try_populate_and_build`] performs at the start; callers that
    /// drive `try_solve_once` directly must call this first.
    ///
    /// [`try_populate_and_build`]: Self::try_populate_and_build
    pub(crate) fn init_shards_and_seed(&mut self) -> u64 {
        if let Some(expected_num_keys) = self.expected_num_keys {
            self.shard_edge.set_up_shards(expected_num_keys, self.eps);
            self.log2_buckets = self.shard_edge.shard_high_bits();
        }
        self.seed
    }

    /// Copies behavioral configuration from another builder into `self`,
    /// regardless of the other builder's type parameters.
    ///
    /// The copied fields are: [`max_num_threads`], [`offline`],
    /// [`check_dups`], [`low_mem`], and [`eps`]. Data-dependent fields
    /// ([`expected_num_keys`], [`seed`], [`log2_buckets`]) and internal
    /// construction state are left at their defaults.
    ///
    /// [`max_num_threads`]: Self::max_num_threads
    /// [`offline`]: Self::offline
    /// [`check_dups`]: Self::check_dups
    /// [`low_mem`]: Self::low_mem
    /// [`eps`]: Self::eps
    /// [`expected_num_keys`]: Self::expected_num_keys
    /// [`seed`]: Self::seed
    /// [`log2_buckets`]: Self::log2_buckets
    pub fn set_from<D2, S2, E2>(mut self, other: &VBuilder<D2, S2, E2>) -> Self {
        self.max_num_threads = other.max_num_threads;
        self.offline = other.offline;
        self.check_dups = other.check_dups;
        self.low_mem = other.low_mem;
        self.eps = other.eps;
        self
    }
}

/// Fatal build errors.
#[derive(thiserror::Error, Debug)]
pub enum BuildError {
    /// A duplicate key was detected with high probability (multiple
    /// construction attempts with different seeds found duplicate 128-bit
    /// signatures).
    #[error("Duplicate key")]
    DuplicateKey,
    /// Multiple construction attempts found duplicate local signatures.
    #[error("Duplicate local signatures: use full signatures")]
    DuplicateLocalSignatures,
    /// A value is too large for the specified bit size.
    #[error("Value too large for specified bit size")]
    ValueTooLarge,
    /// All shards remained unsolvable after multiple attempts.
    #[error("Unsolvable shard after multiple attempts")]
    UnsolvableShard,
}

/// Transient error during the build, leading to trying with a different seed.
#[derive(thiserror::Error, Debug)]
pub enum SolveError {
    /// A duplicate signature was detected.
    #[error("Duplicate signature")]
    DuplicateSignature,
    /// A duplicate local signature was detected.
    #[error("Duplicate local signature")]
    DuplicateLocalSignature,
    /// The maximum shard is too big relative to the average shard.
    #[error("Max shard too big")]
    MaxShardTooBig,
    /// A shard cannot be solved.
    #[error("Unsolvable shard")]
    UnsolvableShard,
}

/// The result of a peeling procedure.
enum PeelResult<
    'a,
    D: BitFieldSlice<Value: Word + BinSafe + Send + Sync> + BitFieldSliceMut + Send + Sync + 'a,
    S: Sig + BinSafe,
    E: ShardEdge<S, 3>,
    V: BinSafe,
> {
    Complete(),
    Partial {
        /// The shard index.
        shard_index: usize,
        /// The shard.
        shard: Arc<Vec<SigVal<S, V>>>,
        /// The data.
        data: ShardData<'a, D>,
        /// The double stack whose upper stack contains the peeled edges
        /// (possibly represented by the vertex from which they have been
        /// peeled).
        double_stack: DoubleStack<E::Vertex>,
        /// The sides stack.
        sides_stack: Vec<u8>,
    },
}

/// An iterator over segments of data associated with each shard.
type ShardDataIter<'a, D> = <D as SliceByValueMut>::ChunksMut<'a>;
/// A segment of data associated with a specific shard.
type ShardData<'a, D> = <ShardDataIter<'a, D> as Iterator>::Item;

impl<W: Word + BinSafe, S: Sig + Send + Sync, E: ShardEdge<S, 3> + MemSize + FlatType>
    VBuilder<BitFieldVec<Box<[W]>>, S, E>
where
    BitFieldVec<Box<[W]>>: MemSize + FlatType,
    SigVal<S, W>: RadixKey,
    SigVal<E::LocalSig, W>: BitXor + BitXorAssign,
{
    /// Builds a new function by reusing an existing [`ShardStore`] with
    /// remapped values.
    ///
    /// This avoids re-hashing the keys: the store already contains the
    /// signatures from a previous build. A `get_val` closure transforms
    /// each stored [`SigVal`] into the new output value.
    ///
    /// # Preconditions
    ///
    /// * `seed` and `shard_edge` **must** be the exact values used when
    ///   the store was populated; passing mismatched values produces a
    ///   silently corrupt function.
    /// * Every value produced by `get_val` must fit in the bit width
    ///   implied by `max_value`.
    /// * `get_val` must be deterministic: the store is iterated multiple
    ///   times and different results for the same input would silently
    ///   corrupt the function.
    pub fn try_build_func_with_store<K: ?Sized + ToSig<S>, V: BinSafe + Default + Send + Sync>(
        &mut self,
        seed: u64,
        shard_edge: E,
        max_value: W,
        shard_store: &mut (impl ShardStore<S, V> + ?Sized),
        get_val: &(impl Fn(&E, SigVal<E::LocalSig, V>) -> W + Send + Sync),
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> anyhow::Result<VFunc<K, BitFieldVec<Box<[W]>>, S, E>>
    where
        SigVal<S, V>: RadixKey,
        SigVal<E::LocalSig, V>: BitXor + BitXorAssign,
        for<'a> ShardDataIter<'a, BitFieldVec<Box<[W]>>>: Send,
        for<'a> <ShardDataIter<'a, BitFieldVec<Box<[W]>>> as Iterator>::Item: Send,
    {
        self.try_build_func_with_store_and_inspect(
            seed,
            shard_edge,
            max_value,
            shard_store,
            get_val,
            &|_| {},
            pl,
        )
    }

    /// Like [`try_build_func_with_store`], but calls `inspect` on each
    /// [`SigVal`] during the peeling phase.
    ///
    /// This is used by [`VFunc2`] and [`Lcp2MmphfInt`] to count escaped
    /// keys per shard without a separate pass over the store.
    ///
    /// [`try_build_func_with_store`]: Self::try_build_func_with_store
    /// [`VFunc2`]: crate::func::VFunc2
    /// [`Lcp2MmphfInt`]: crate::func::Lcp2MmphfInt
    pub fn try_build_func_with_store_and_inspect<
        K: ?Sized + ToSig<S>,
        V: BinSafe + Default + Send + Sync,
    >(
        &mut self,
        seed: u64,
        shard_edge: E,
        max_value: W,
        shard_store: &mut (impl ShardStore<S, V> + ?Sized),
        get_val: &(impl Fn(&E, SigVal<E::LocalSig, V>) -> W + Send + Sync),
        inspect: &(impl Fn(&SigVal<S, V>) + Send + Sync),
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> anyhow::Result<VFunc<K, BitFieldVec<Box<[W]>>, S, E>>
    where
        SigVal<S, V>: RadixKey,
        SigVal<E::LocalSig, V>: BitXor + BitXorAssign,
        for<'a> ShardDataIter<'a, BitFieldVec<Box<[W]>>>: Send,
        for<'a> <ShardDataIter<'a, BitFieldVec<Box<[W]>>> as Iterator>::Item: Send,
    {
        self.shard_edge = shard_edge;
        self.num_keys = shard_store.len();
        self.bit_width = max_value.bit_len() as usize;

        // The shard structure is fixed by the store (set at `into_shard_store`
        // time). We only reconfigure graph parameters (c, lge, segment size,
        // vertex count) for the actual key count and shard sizes.
        let max_shard = shard_store.shard_sizes().max().unwrap_or(0);
        (self.c, self.lge) = self.shard_edge.set_up_graphs(self.num_keys, max_shard);

        pl.info(format_args!("Max value: {max_value}"));

        let data: BitFieldVec<Box<[W]>> = BitFieldVec::<Box<[W]>>::new_padded(
            self.bit_width,
            self.shard_edge.num_vertices() * self.shard_edge.num_shards(),
        );

        self.try_build_from_shard_iter(seed, data, shard_store.iter(), get_val, inspect, pl)
            .map_err(Into::into)
    }
}

/// State for the retry loop used by
/// [`VBuilder::try_populate_and_build`] and external callers that drive
/// [`VBuilder::try_solve_once`] directly.
///
/// Create with [`VBuilder::retry_state`], then call
/// [`handle_solve_result`] after each attempt.
///
/// [`handle_solve_result`]: RetryState::handle_solve_result
pub(crate) struct RetryState {
    prng: SmallRng,
    dup_count: u32,
    local_dup_count: u32,
    unsolvable_count: u32,
}

impl RetryState {
    /// Returns the next seed to try.
    pub(crate) fn next_seed(&mut self) -> u64 {
        self.prng.random()
    }

    /// Handles the result of a [`VBuilder::try_solve_once`] call.
    ///
    /// On success, returns `Ok(Some(r))`. On retryable error, logs a
    /// warning and returns `Ok(None)` (the caller should rewind its
    /// lenders and retry). On fatal error, returns `Err(e)`.
    pub(crate) fn handle_solve_result<R>(
        &mut self,
        result: anyhow::Result<R>,
        pl: &mut impl ProgressLog,
    ) -> anyhow::Result<Option<R>> {
        match result {
            Ok(r) => Ok(Some(r)),
            Err(error) => match error.downcast::<SolveError>() {
                Ok(vfunc_error) => match vfunc_error {
                    SolveError::DuplicateSignature => {
                        if self.dup_count >= 3 {
                            pl.error(format_args!("Duplicate keys (duplicate 128-bit signatures with four different seeds)"));
                            return Err(BuildError::DuplicateKey.into());
                        }
                        pl.warn(format_args!(
                            "Duplicate 128-bit signature, trying again with a different seed..."
                        ));
                        self.dup_count += 1;
                        Ok(None)
                    }
                    SolveError::DuplicateLocalSignature => {
                        if self.local_dup_count >= 2 {
                            pl.error(format_args!("Duplicate local signatures: use full signatures (duplicate local signatures with three different seeds)"));
                            return Err(BuildError::DuplicateLocalSignatures.into());
                        }
                        pl.warn(format_args!(
                            "Duplicate local signature, trying again with a different seed..."
                        ));
                        self.local_dup_count += 1;
                        Ok(None)
                    }
                    SolveError::MaxShardTooBig => {
                        pl.warn(format_args!(
                            "The maximum shard is too big, trying again with a different seed..."
                        ));
                        Ok(None)
                    }
                    SolveError::UnsolvableShard => {
                        self.unsolvable_count += 1;
                        if self.unsolvable_count >= 100 {
                            panic!("Failed more than 100 attempts (this shouldn't happen)");
                        }
                        pl.info(format_args!(
                            "Unsolvable shard, trying again with a different seed..."
                        ));
                        Ok(None)
                    }
                },
                Err(error) => Err(error),
            },
        }
    }
}

impl<
    D: BitFieldSlice<Value: Word + BinSafe> + Send + Sync,
    S: Sig + Send + Sync,
    E: ShardEdge<S, 3>,
> VBuilder<D, S, E>
{
    /// Builds a new [`VFunc`], draining the shard store during
    /// construction to free memory as shards are consumed.
    ///
    /// This is a low-level method that requires a thorough understanding
    /// of the builder's internal state.
    ///
    /// `new_data(bit_width, len)` allocates the backend storage of the
    /// given bit width and length.
    ///
    /// The store is drained (freed) during construction. If you need
    /// to keep the store for building signed wrappers or secondary
    /// structures, use [`try_build_func_and_store`] instead.
    ///
    /// [`try_build_func_and_store`]: Self::try_build_func_and_store
    pub(crate) fn try_build_func<
        K: ?Sized + ToSig<S> + std::fmt::Debug,
        B: ?Sized + Borrow<K>,
        P: ProgressLog + Clone + Send + Sync,
        L: FallibleRewindableLender<
                RewindError: Error + Send + Sync + 'static,
                Error: Error + Send + Sync + 'static,
            > + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
    >(
        mut self,
        keys: L,
        values: impl FallibleRewindableLender<
            RewindError: Error + Send + Sync + 'static,
            Error: Error + Send + Sync + 'static,
        > + for<'lend> FallibleLending<'lend, Lend = &'lend D::Value>,
        new_data: fn(usize, usize) -> D,
        pl: &mut P,
    ) -> anyhow::Result<(VFunc<K, D, S, E>, L)>
    where
        SigVal<S, D::Value>: RadixKey,
        SigVal<E::LocalSig, D::Value>: BitXor + BitXorAssign,
        D: for<'a> BitFieldSliceMut<ChunksMut<'a>: Iterator<Item: BitFieldSliceMut>>
            + MemSize
            + FlatType,
        E: MemSize + FlatType,
        for<'a> ShardDataIter<'a, D>: Send,
        for<'a> <ShardDataIter<'a, D> as Iterator>::Item: Send,
    {
        let get_val = |_shard_edge: &E, sig_val: SigVal<E::LocalSig, D::Value>| sig_val.val;

        self.try_populate_and_build(
            keys,
            values,
            &mut |builder, seed, mut store, max_value, _num_keys, pl: &mut P, _state: &mut ()| {
                builder.bit_width = max_value.bit_len() as usize;

                let data = new_data(
                    builder.bit_width,
                    builder.shard_edge.num_vertices() * builder.shard_edge.num_shards(),
                );

                pl.info(format_args!("Max value: {max_value}",));

                let func = builder.try_build_from_shard_iter(
                    seed,
                    data,
                    store.drain(),
                    &get_val,
                    &|_| {},
                    pl,
                )?;
                Ok(func)
            },
            pl,
            (),
        )
    }

    /// Builds a new [`VFunc`], preserving the populated shard store for
    /// reuse.
    ///
    /// This is a low-level method that requires a thorough understanding
    /// of the builder's internal state.
    ///
    /// `new_data(bit_width, len)` allocates the backend storage of the
    /// given bit width and length.
    ///
    /// The second element of the returned tuple is the store that was
    /// populated during construction. The caller can pass it to
    /// [`try_build_func_with_store`] to build additional functions
    /// without re-hashing the keys, or simply drop it. The store is
    /// preserved intact (not drained). If you do not need the store,
    /// use [`try_build_func`] instead, which drains the store to free
    /// memory during construction.
    ///
    /// [`try_build_func_with_store`]: Self::try_build_func_with_store
    /// [`try_build_func`]: Self::try_build_func
    pub(crate) fn try_build_func_and_store<
        K: ?Sized + ToSig<S> + std::fmt::Debug,
        B: ?Sized + Borrow<K>,
        P: ProgressLog + Clone + Send + Sync,
        L: FallibleRewindableLender<
                RewindError: Error + Send + Sync + 'static,
                Error: Error + Send + Sync + 'static,
            > + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
    >(
        mut self,
        keys: L,
        values: impl FallibleRewindableLender<
            RewindError: Error + Send + Sync + 'static,
            Error: Error + Send + Sync + 'static,
        > + for<'lend> FallibleLending<'lend, Lend = &'lend D::Value>,
        new_data: fn(usize, usize) -> D,
        pl: &mut P,
    ) -> anyhow::Result<(
        VFunc<K, D, S, E>,
        Box<dyn ShardStore<S, D::Value> + Send + Sync>,
        L,
    )>
    where
        SigVal<S, D::Value>: RadixKey,
        SigVal<E::LocalSig, D::Value>: BitXor + BitXorAssign,
        D: for<'a> BitFieldSliceMut<ChunksMut<'a>: Iterator<Item: BitFieldSliceMut>>
            + MemSize
            + FlatType,
        E: MemSize + FlatType,
        for<'a> ShardDataIter<'a, D>: Send,
        for<'a> <ShardDataIter<'a, D> as Iterator>::Item: Send,
    {
        let get_val = |_shard_edge: &E, sig_val: SigVal<E::LocalSig, D::Value>| sig_val.val;

        self.try_populate_and_build(
            keys,
            values,
            &mut |builder, seed, mut store, max_value, _num_keys, pl: &mut P, _state: &mut ()| {
                builder.bit_width = max_value.bit_len() as usize;

                let data = new_data(
                    builder.bit_width,
                    builder.shard_edge.num_vertices() * builder.shard_edge.num_shards(),
                );

                pl.info(format_args!("Max value: {max_value}"));

                let func = builder.try_build_from_shard_iter(
                    seed,
                    data,
                    store.iter(),
                    &get_val,
                    &|_| {},
                    pl,
                )?;
                Ok((func, store))
            },
            pl,
            (),
        )
        .map(|((func, store), keys)| (func, store, keys))
    }

    /// Builds a [`VFunc`] suitable for use as a filter backend.
    ///
    /// Unlike [`try_build_func_and_store`], this method takes only keys
    /// (no values), uses a caller-specified `bit_width` instead of
    /// deriving it from `max_value`, and derives stored values from
    /// signatures via `get_val`.
    ///
    /// [`try_build_func_and_store`]: Self::try_build_func_and_store
    ///
    /// `new_data(bit_width, len)` allocates the backend storage.
    pub(crate) fn try_build_filter<
        K: ?Sized + ToSig<S> + std::fmt::Debug,
        B: ?Sized + Borrow<K>,
        P: ProgressLog + Clone + Send + Sync,
        G: Fn(&E, SigVal<E::LocalSig, EmptyVal>) -> D::Value + Send + Sync,
    >(
        mut self,
        mut keys: impl FallibleRewindableLender<
            RewindError: Error + Send + Sync + 'static,
            Error: Error + Send + Sync + 'static,
        > + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
        bit_width: usize,
        new_data: fn(usize, usize) -> D,
        get_val: &G,
        pl: &mut P,
    ) -> anyhow::Result<VFunc<K, D, S, E>>
    where
        SigVal<S, EmptyVal>: RadixKey,
        SigVal<E::LocalSig, EmptyVal>: BitXor + BitXorAssign,
        D: for<'a> BitFieldSliceMut<ChunksMut<'a>: Iterator<Item: BitFieldSliceMut>>
            + MemSize
            + FlatType,
        E: MemSize + FlatType,
        for<'a> ShardDataIter<'a, D>: Send,
        for<'a> <ShardDataIter<'a, D> as Iterator>::Item: Send,
    {
        let mut rs = self.retry_state(pl);

        loop {
            let seed = rs.next_seed();

            let result = {
                let mut populate =
                    |seed: u64,
                     push: &mut dyn FnMut(SigVal<S, EmptyVal>) -> anyhow::Result<()>,
                     pl: &mut P,
                     _state: &mut ()| {
                        while let Some(key) = keys.next()? {
                            pl.light_update();
                            push(SigVal {
                                sig: K::to_sig(key.borrow(), seed),
                                val: EmptyVal::default(),
                            })?;
                        }
                        Ok(EmptyVal::default())
                    };

                self.try_solve_once(
                    seed,
                    &mut populate,
                    &mut |builder,
                          seed,
                          mut store,
                          _max_value,
                          _num_keys,
                          pl: &mut P,
                          _state: &mut ()| {
                        builder.bit_width = bit_width;

                        let data = new_data(
                            builder.bit_width,
                            builder.shard_edge.num_vertices() * builder.shard_edge.num_shards(),
                        );

                        let func = builder.try_build_from_shard_iter(
                            seed,
                            data,
                            store.drain(),
                            get_val,
                            &|_| {},
                            pl,
                        )?;
                        Ok(func)
                    },
                    pl,
                    &mut (),
                )
            };

            if let Some(r) = rs.handle_solve_result(result, pl)? {
                return Ok(r);
            }

            keys = keys.rewind()?;
        }
    }

    /// Initializes shards and returns a [`RetryState`] for driving the
    /// retry loop.
    ///
    /// This is the same initialization that
    /// [`try_populate_and_build`] performs at the start.
    ///
    /// [`try_populate_and_build`]: Self::try_populate_and_build
    pub(crate) fn retry_state(&mut self, pl: &mut impl ProgressLog) -> RetryState {
        self.init_shards_and_seed();
        pl.info(format_args!(
            "Using a store with 2^{} buckets",
            self.log2_buckets
        ));
        RetryState {
            prng: SmallRng::seed_from_u64(self.seed),
            dup_count: 0,
            local_dup_count: 0,
            unsolvable_count: 0,
        }
    }

    /// Populates a shard store from keys and values, then calls a build
    /// closure. If the closure (or the store population) fails with a
    /// [`SolveError`], the process is retried from scratch with a new
    /// seed.
    ///
    /// This is a low-level method that requires a thorough understanding
    /// of the builder's internal state.
    ///
    /// On each retry the lenders are rewound. Retries continue for
    /// [`SolveError::UnsolvableShard`] and
    /// [`SolveError::MaxShardTooBig`]; after 4 duplicate-signature
    /// retries [`BuildError::DuplicateKey`] is returned, and after 3
    /// duplicate-local-signature retries
    /// [`BuildError::DuplicateLocalSignatures`] is returned. Any other
    /// error is propagated immediately.
    ///
    /// `build_fn` is called with `(&mut self, seed, store, max_value, num_keys,
    /// pl)`. The builder's `shard_edge`, `c`, and `lge` fields are already set
    /// up when `build_fn` is invoked, so it can call
    /// `try_build_from_shard_iter` directly.
    ///
    /// Returns whatever `build_fn` returns on success.
    pub(crate) fn try_populate_and_build<
        K: ?Sized + ToSig<S> + std::fmt::Debug,
        B: ?Sized + Borrow<K>,
        V: BinSafe + Default + Send + Sync + Ord,
        R,
        P: ProgressLog + Clone + Send + Sync,
        C,
        L: FallibleRewindableLender<
                RewindError: Error + Send + Sync + 'static,
                Error: Error + Send + Sync + 'static,
            > + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
    >(
        &mut self,
        mut keys: L,
        mut values: impl FallibleRewindableLender<
            RewindError: Error + Send + Sync + 'static,
            Error: Error + Send + Sync + 'static,
        > + for<'lend> FallibleLending<'lend, Lend = &'lend V>,
        build_fn: &mut impl FnMut(
            &mut Self,
            u64,
            Box<dyn ShardStore<S, V> + Send + Sync>,
            V,
            usize,
            &mut P,
            &mut C,
        ) -> anyhow::Result<R>,
        pl: &mut P,
        mut state: C,
    ) -> anyhow::Result<(R, L)>
    where
        SigVal<S, V>: RadixKey,
    {
        let mut rs = self.retry_state(pl);
        let total_start = Instant::now();

        loop {
            let seed = rs.next_seed();

            let result = {
                let mut populate = |seed: u64,
                                    push: &mut dyn FnMut(SigVal<S, V>) -> anyhow::Result<()>,
                                    pl: &mut P,
                                    _state: &mut C| {
                    let mut maybe_max_value = V::default();
                    while let Some(key) = keys.next()? {
                        pl.light_update();
                        let &maybe_val = values.next()?.expect("Not enough values");
                        maybe_max_value = Ord::max(maybe_max_value, maybe_val);
                        push(SigVal {
                            sig: K::to_sig(key.borrow(), seed),
                            val: maybe_val,
                        })?;
                    }
                    Ok(maybe_max_value)
                };

                self.try_solve_once(seed, &mut populate, build_fn, pl, &mut state)
            };

            if let Some(r) = rs.handle_solve_result(result, pl)? {
                let num_keys = self.num_keys;
                pl.info(format_args!(
                    "Construction completed in {:.3} seconds ({} keys, {:.3} ns/key)",
                    total_start.elapsed().as_secs_f64(),
                    num_keys,
                    total_start.elapsed().as_nanos() as f64 / num_keys as f64
                ));
                return Ok((r, keys));
            }

            values = values.rewind()?;
            keys = keys.rewind()?;
        }
    }

    /// Like [`try_populate_and_build`], but takes key and value
    /// **slices** and parallelizes the hash computation and store
    /// population using rayon.
    ///
    /// [`try_populate_and_build`]: Self::try_populate_and_build
    ///
    /// Each key is hashed on a rayon worker thread and deposited directly
    /// into its SigStore bucket (protected by a per-bucket mutex). This is
    /// typically faster than the lender-based path for large in-memory key
    /// sets, because the expensive per-key hashing runs on all available
    /// cores.
    pub(crate) fn try_par_populate_and_build<
        K: ?Sized + ToSig<S> + std::fmt::Debug + Sync,
        B: Borrow<K> + Sync,
        V: BinSafe + Default + Send + Sync + Ord + Copy,
        R,
        P: ProgressLog + Clone + Send + Sync,
        C,
        VF: Fn(usize) -> V + Send + Sync,
    >(
        &mut self,
        keys: &[B],
        val_fn: &VF,
        build_fn: &mut impl FnMut(
            &mut Self,
            u64,
            Box<dyn ShardStore<S, V> + Send + Sync>,
            V,
            usize,
            &mut P,
            &mut C,
        ) -> anyhow::Result<R>,
        pl: &mut P,
        mut state: C,
    ) -> anyhow::Result<R>
    where
        SigVal<S, V>: RadixKey,
        S: Send,
    {
        let mut rs = self.retry_state(pl);
        let n = keys.len();
        let total_start = Instant::now();

        loop {
            let seed = rs.next_seed();

            let result = {
                let mut sig_store = sig_store::new_online::<S, V>(
                    self.log2_buckets,
                    LOG2_MAX_SHARDS,
                    self.expected_num_keys,
                )?;

                pl.expected_updates(n);
                pl.item_name("key");
                pl.start(format!(
                    "Computing and storing {}-bit signatures in parallel in RAM using seed 0x{seed:016x}...",
                    std::mem::size_of::<S>() * 8,
                ));

                let start = Instant::now();

                let maybe_max_value = sig_store.par_populate(n, self.max_num_threads, |i| SigVal {
                    sig: K::to_sig(keys[i].borrow(), seed),
                    val: val_fn(i),
                });

                pl.done();

                let num_keys = sig_store.len();

                pl.info(format_args!(
                    "Computation of signatures from inputs completed in {:.3} seconds ({} keys, {:.3} ns/key)",
                    start.elapsed().as_secs_f64(),
                    num_keys,
                    start.elapsed().as_nanos() as f64 / num_keys as f64
                ));
                // When `shard_size_hint` is set (e.g. by CompVFunc with
                // `total_edges` after building the Huffman coder), the
                // sharding strategy is sized for the hinted workload
                // rather than the raw key count. See the doc comment on
                // [`VBuilder::shard_size_hint`].
                let shard_n = self.shard_size_hint.unwrap_or(num_keys);
                let shard_edge = &mut self.shard_edge;
                shard_edge.set_up_shards(shard_n, self.eps);

                let shard_store = sig_store.into_shard_store(shard_edge.shard_high_bits())?;
                let max_shard = shard_store.shard_sizes().max().unwrap_or(0);

                if self.shard_size_hint.is_none()
                    && max_shard as f64 > 1.01 * num_keys as f64 / shard_edge.num_shards() as f64
                {
                    Err(SolveError::MaxShardTooBig.into())
                } else {
                    (self.c, self.lge) = shard_edge.set_up_graphs(num_keys, max_shard);
                    self.num_keys = num_keys;
                    let store = Box::new(shard_store) as Box<dyn ShardStore<S, V> + Send + Sync>;
                    build_fn(self, seed, store, maybe_max_value, num_keys, pl, &mut state)
                }
            };

            if let Some(r) = rs.handle_solve_result(result, pl)? {
                let num_keys = self.num_keys;
                pl.info(format_args!(
                    "Construction completed in {:.3} seconds ({} keys, {:.3} ns/key)",
                    total_start.elapsed().as_secs_f64(),
                    num_keys,
                    total_start.elapsed().as_nanos() as f64 / num_keys as f64
                ));
                return Ok(r);
            }
            // Keys and values are slices — no rewind needed.
        }
    }

    /// Performs a single population-and-build attempt for the given seed.
    ///
    /// Creates a [`SigStore`] (offline or online depending on
    /// `self.offline`), populates it via the `populate` closure, converts
    /// it to a shard store, checks the max-shard invariant, sets up
    /// `self.shard_edge`/`c`/`lge`, boxes the store into a
    /// `dyn` [`ShardStore`], and calls `build_fn`.
    ///
    /// The `populate` closure receives `(seed, push, &mut pl, &mut state)`
    /// where `push` appends a [`SigVal`] to the store. The closure must
    /// iterate the keys, push all signature/value pairs, and return the
    /// maximum value seen. The `build_fn` closure receives the same
    /// `&mut state` as its last argument.
    ///
    /// `state` is a caller-supplied context of type `C` that is threaded
    /// through both closures, allowing them to share mutable data without
    /// interior mutability. Pass `&mut ()` when no shared state is needed.
    ///
    /// Returns the result of `build_fn`, or a [`SolveError`] for the
    /// caller's retry loop to handle.
    pub(crate) fn try_solve_once<
        V: BinSafe + Default + Send + Sync + Ord,
        R,
        P: ProgressLog + Clone + Send + Sync,
        C,
    >(
        &mut self,
        seed: u64,
        populate: &mut impl FnMut(
            u64,
            &mut dyn FnMut(SigVal<S, V>) -> anyhow::Result<()>,
            &mut P,
            &mut C,
        ) -> anyhow::Result<V>,
        build_fn: &mut impl FnMut(
            &mut Self,
            u64,
            Box<dyn ShardStore<S, V> + Send + Sync>,
            V,
            usize,
            &mut P,
            &mut C,
        ) -> anyhow::Result<R>,
        pl: &mut P,
        state: &mut C,
    ) -> anyhow::Result<R>
    where
        SigVal<S, V>: RadixKey,
    {
        if self.offline {
            self.try_solve_once_inner(
                seed,
                sig_store::new_offline::<S, V>(
                    self.log2_buckets,
                    LOG2_MAX_SHARDS,
                    self.expected_num_keys,
                )?,
                populate,
                build_fn,
                pl,
                state,
            )
        } else {
            self.try_solve_once_inner(
                seed,
                sig_store::new_online::<S, V>(
                    self.log2_buckets,
                    LOG2_MAX_SHARDS,
                    self.expected_num_keys,
                )?,
                populate,
                build_fn,
                pl,
                state,
            )
        }
    }

    /// Inner generic implementation of [`try_solve_once`].
    ///
    /// [`try_solve_once`]: Self::try_solve_once
    ///
    /// This is generic over `SS` so that the `populate` closure can push
    /// to the concrete store type without dynamic dispatch. The `state`
    /// parameter is forwarded to both `populate` and `build_fn`.
    fn try_solve_once_inner<
        V: BinSafe + Default + Send + Sync + Ord,
        R,
        P: ProgressLog + Clone + Send + Sync,
        SS: SigStore<S, V, ShardStore: 'static>,
        C,
    >(
        &mut self,
        seed: u64,
        mut sig_store: SS,
        populate: &mut impl FnMut(
            u64,
            &mut dyn FnMut(SigVal<S, V>) -> anyhow::Result<()>,
            &mut P,
            &mut C,
        ) -> anyhow::Result<V>,
        build_fn: &mut impl FnMut(
            &mut Self,
            u64,
            Box<dyn ShardStore<S, V> + Send + Sync>,
            V,
            usize,
            &mut P,
            &mut C,
        ) -> anyhow::Result<R>,
        pl: &mut P,
        state: &mut C,
    ) -> anyhow::Result<R>
    where
        SigVal<S, V>: RadixKey,
    {
        pl.expected_updates(self.expected_num_keys);
        pl.item_name("key");
        pl.start(format!(
            "Computing and storing {}-bit signatures sequentially {} using seed 0x{:016x}...",
            std::mem::size_of::<S>() * 8,
            sig_store
                .temp_dir()
                .map(|d| Cow::Owned(format!("on disk ({})", d.path().display())))
                .unwrap_or(Cow::Borrowed("in RAM")),
            seed
        ));

        let start = Instant::now();

        let maybe_max_value = populate(
            seed,
            &mut |sig_val| sig_store.try_push(sig_val).map_err(Into::into),
            pl,
            state,
        )?;

        pl.done();

        let num_keys = sig_store.len();

        pl.info(format_args!(
            "Computation of signatures from inputs completed in {:.3} seconds ({} keys, {:.3} ns/key)",
            start.elapsed().as_secs_f64(),
            num_keys,
            start.elapsed().as_nanos() as f64 / num_keys as f64
        ));

        // See the doc comment on [`VBuilder::shard_size_hint`]:
        // when set, use it to size the sharding strategy instead of
        // the runtime key count. Used by CompVFunc to pass the total
        // edge count (≈ entropy × num_keys).
        let shard_n = self.shard_size_hint.unwrap_or(num_keys);
        let shard_edge = &mut self.shard_edge;
        shard_edge.set_up_shards(shard_n, self.eps);

        let shard_store = sig_store.into_shard_store(shard_edge.shard_high_bits())?;
        let max_shard = shard_store.shard_sizes().max().unwrap_or(0);

        if shard_edge.shard_high_bits() != 0 {
            pl.info(format_args!(
                "Max shard / average shard: {:.3}%",
                (100.0 * max_shard as f64) / (num_keys as f64 / shard_edge.num_shards() as f64)
            ));
        }

        if self.shard_size_hint.is_none()
            && max_shard as f64 > 1.01 * num_keys as f64 / shard_edge.num_shards() as f64
        {
            return Err(SolveError::MaxShardTooBig.into());
        }

        (self.c, self.lge) = shard_edge.set_up_graphs(num_keys, max_shard);
        self.num_keys = num_keys;

        let store = Box::new(shard_store) as Box<dyn ShardStore<S, V> + Send + Sync>;

        build_fn(self, seed, store, maybe_max_value, num_keys, pl, state)
    }
}

impl<
    D: BitFieldSlice<Value: Word + BinSafe>
        + for<'a> BitFieldSliceMut<ChunksMut<'a>: Iterator<Item: BitFieldSliceMut>>
        + Send
        + Sync
        + MemSize
        + FlatType,
    S: Sig + Send + Sync,
    E: ShardEdge<S, 3> + MemSize + FlatType,
> VBuilder<D, S, E>
{
    /// Solves the 3-hypergraph system and returns a new [`VFunc`].
    ///
    /// This is the core solver: it peels the hypergraph defined by the
    /// shard iterator, writes the solution into `data`, and returns the
    /// assembled [`VFunc`].
    ///
    /// # Preconditions
    ///
    /// * `seed` must be the seed used during store population.
    ///
    /// * `data` must be freshly allocated, zero-initialized storage of
    ///   size `shard_edge.num_vertices() * shard_edge.num_shards()`.
    ///
    /// * `self.shard_edge`, `self.c`, `self.lge`, `self.bit_width`, and
    ///   `self.num_keys` must be set up by the caller (typically by
    ///   [`try_solve_once`]).
    ///
    /// [`try_solve_once`]: Self::try_solve_once
    ///
    /// * `get_val` must be deterministic.
    ///
    /// # Errors
    ///
    /// Returns [`SolveError::UnsolvableShard`] or
    /// [`SolveError::DuplicateLocalSignature`] if the system cannot be
    /// solved with the current seed.
    ///
    /// The peeling algorithm is selected based on `self.lge` and
    /// `self.low_mem`; see the [`low_mem`] field documentation for the
    /// automatic selection heuristic.
    ///
    /// [`low_mem`]: VBuilder::low_mem
    pub(crate) fn try_build_from_shard_iter<
        K: ?Sized + ToSig<S>,
        I,
        P,
        V: BinSafe + Default + Send + Sync,
        G: Fn(&E, SigVal<E::LocalSig, V>) -> D::Value + Send + Sync,
        H: Fn(&SigVal<S, V>) + Send + Sync,
    >(
        &mut self,
        seed: u64,
        mut data: D,
        shard_iter: I,
        get_val: &G,
        inspect: &H,
        pl: &mut P,
    ) -> Result<VFunc<K, D, S, E>, SolveError>
    where
        SigVal<S, V>: RadixKey,
        SigVal<E::LocalSig, V>: BitXor + BitXorAssign,
        P: ProgressLog + Clone + Send + Sync,
        I: Iterator<Item = Arc<Vec<SigVal<S, V>>>> + Send,
        for<'a> ShardDataIter<'a, D>: Send,
        for<'a> <ShardDataIter<'a, D> as Iterator>::Item: Send,
    {
        // Static check that Vertex → usize conversion is lossless
        const {
            assert!(
                size_of::<E::Vertex>() <= size_of::<usize>(),
                "ShardEdge::Vertex must fit in usize without truncation"
            );
        }

        let shard_edge = &self.shard_edge;
        self.num_threads = shard_edge.num_shards().min(self.max_num_threads);

        pl.info(format_args!(
            "Number of keys: {}; bit width: {}",
            self.num_keys, self.bit_width,
        ));

        pl.info(format_args!(
            "{}; signatures: {}",
            shard_edge,
            core::any::type_name::<S>()
        ));

        pl.info(format_args!(
            "c: {}; overhead: {:+.3}%; threads: {}",
            self.c,
            100. * ((shard_edge.num_vertices() * shard_edge.num_shards()) as f64
                / (self.num_keys as f64)
                - 1.),
            self.num_threads
        ));

        // main_pl (shard counter) stays at info; per-shard detail at trace
        let mut main_pl = pl.concurrent();
        pl.log_level(log::Level::Trace);

        if self.lge {
            self.par_solve(
                shard_iter,
                &mut data,
                0,
                |this, shard_index, shard, data, pl| {
                    this.lge_shard(shard_index, shard, data, get_val, inspect, pl)
                },
                &mut main_pl,
                pl,
            )?;
        } else if self.low_mem == Some(true)
            || self.low_mem.is_none() && self.num_threads > 3 && shard_edge.num_shards() > 2
        {
            self.par_solve(
                shard_iter,
                &mut data,
                0,
                |this, shard_index, shard, data, pl| {
                    this.peel_by_sig_vals_low_mem(shard_index, shard, data, get_val, inspect, pl)
                },
                &mut main_pl,
                pl,
            )?;
        } else {
            self.par_solve(
                shard_iter,
                &mut data,
                0,
                |this, shard_index, shard, data, pl| {
                    this.peel_by_sig_vals_high_mem(shard_index, shard, data, get_val, inspect, pl)
                },
                &mut main_pl,
                pl,
            )?;
        }

        pl.log_level(log::Level::Info);

        let num_keys = self.num_keys;
        Ok(VFunc {
            seed,
            shard_edge: self.shard_edge,
            num_keys,
            data,
            _marker: std::marker::PhantomData,
        })
        .inspect(|result| {
            let bit_size = result.mem_size(SizeFlags::default()) as f64 * 8.0;
            pl.info(format_args!(
                "Bits/key: {:.3} ({:+.3}% with respect to bit width)",
                bit_size / num_keys as f64,
                100.0 * (bit_size / (result.data.bit_width() as f64 * num_keys as f64) - 1.),
            ));
        })
    }
}

// `XorGraph`, `DoubleStack`, `FastStack`, and the `remove_edge!`
// macro live in [`crate::func::peeling`] so they can be shared with
// [`crate::func::CompVFunc`]. VBuilder's peeler bodies are otherwise
// unchanged.
use crate::func::peeling::{DoubleStack, FastStack, XorGraph, remove_edge};

impl<
    D: BitFieldSlice<Value: Word + BinSafe + Send + Sync>
        + for<'a> BitFieldSliceMut<ChunksMut<'a>: Iterator<Item: BitFieldSliceMut>>
        + Send
        + Sync,
    S: Sig + BinSafe,
    E: ShardEdge<S, 3>,
> VBuilder<D, S, E>
{
    /// Stable counting sort of `shard` by [`ShardEdge::sort_key`],
    /// improving memory locality before peeling.
    fn count_sort<V: BinSafe>(&self, data: &mut [SigVal<S, V>]) {
        let num_sort_keys = self.shard_edge.num_sort_keys();
        let mut count = vec![0; num_sort_keys];

        let mut copied = Box::new_uninit_slice(data.len());
        for (&sig_val, copy) in data.iter().zip(copied.iter_mut()) {
            count[self.shard_edge.sort_key(sig_val.sig)] += 1;
            copy.write(sig_val);
        }
        // SAFETY: every element was initialized in the loop above.
        let copied = unsafe { copied.assume_init() };

        count.iter_mut().fold(0, |acc, c| {
            let old = *c;
            *c = acc;
            acc + old
        });

        for &sig_val in copied.iter() {
            let key = self.shard_edge.sort_key(sig_val.sig);
            data[count[key]] = sig_val;
            count[key] += 1;
        }
    }

    /// After this number of keys, in the case of filters we remove duplicate
    /// edges.
    #[cfg(target_pointer_width = "64")]
    const MAX_NO_LOCAL_SIG_CHECK: usize = 1 << 33;
    #[cfg(not(target_pointer_width = "64"))]
    const MAX_NO_LOCAL_SIG_CHECK: usize = usize::MAX;

    /// Solves in parallel shards returned by an iterator, storing
    /// the result in `data`.
    ///
    /// # Arguments
    ///
    /// * `shard_iter`: an iterator returning the shards to solve.
    ///
    /// * `data`: the storage for the solution values.
    ///
    /// * `solve_shard`: a method to solve a shard; it takes the shard index,
    ///   the shard, shard-local data, and a progress logger. It may
    ///   fail by returning an error.
    ///
    /// * `main_pl`: the progress logger for the overall computation.
    ///
    /// * `pl`: a progress logger that will be cloned to display the progress of
    ///   a current shard.
    ///
    /// # Errors
    ///
    /// This method returns an error if one of the shards cannot be solved, or
    /// if duplicates are detected.
    pub(crate) fn par_solve<
        'b,
        V: BinSafe,
        I: IntoIterator<IntoIter: Send, Item = Arc<Vec<SigVal<S, V>>>> + Send,
        SS: Fn(&Self, usize, Arc<Vec<SigVal<S, V>>>, ShardData<'b, D>, &mut P) -> Result<(), ()>
            + Send
            + Sync
            + Copy,
        C: ConcurrentProgressLog + Send + Sync,
        P: ProgressLog + Clone + Send + Sync,
    >(
        &self,
        shard_iter: I,
        data: &'b mut D,
        shard_stride_padding: usize,
        solve_shard: SS,
        main_pl: &mut C,
        pl: &mut P,
    ) -> Result<(), SolveError>
    where
        SigVal<S, V>: RadixKey,
        for<'a> ShardDataIter<'a, D>: Send,
        for<'a> <ShardDataIter<'a, D> as Iterator>::Item: Send,
    {
        main_pl
            .item_name("shard")
            .expected_updates(self.shard_edge.num_shards())
            .display_memory(true)
            .start("Solving shards...");

        self.failed.store(false, Ordering::Relaxed);
        let num_shards = self.shard_edge.num_shards();
        let buffer_size = self.num_threads.ilog2() as usize;

        let (err_send, err_recv) = crossbeam_channel::bounded::<_>(self.num_threads);
        let (data_send, data_recv) = crossbeam_channel::bounded::<(
            usize,
            (Arc<Vec<SigVal<S, V>>>, ShardData<'_, D>),
        )>(buffer_size);

        let result = std::thread::scope(|scope| {
            scope.spawn(move || {
                let _ = thread_priority::set_current_thread_priority(ThreadPriority::Max);
                let shard_stride = self.shard_edge.num_vertices() + shard_stride_padding;
                for val in shard_iter
                    .into_iter()
                    .zip(data.try_chunks_mut(shard_stride).unwrap())
                    .enumerate()
                {
                    if data_send.send(val).is_err() {
                        break;
                    }
                }

                drop(data_send);
            });

            for _thread_id in 0..self.num_threads {
                let mut main_pl = main_pl.clone();
                let mut pl = pl.clone();
                let err_send = err_send.clone();
                let data_recv = data_recv.clone();
                scope.spawn(move || {
                    loop {
                        match data_recv.recv() {
                            Err(_) => return,
                            Ok((shard_index, (shard, mut data))) => {
                                if shard.is_empty() {
                                    return;
                                }

                                main_pl.debug(format_args!(
                                    "Analyzing shard {}/{}...",
                                    shard_index + 1,
                                    num_shards
                                ));

                                pl.start(format!(
                                    "Sorting shard {}/{}...",
                                    shard_index + 1,
                                    num_shards
                                ));

                                {
                                    // SAFETY: The Arc has refcount 1: this thread is the
                                    // sole owner after receiving from the channel.
                                    let shard = unsafe {
                                        &mut *(Arc::as_ptr(&shard) as *mut Vec<SigVal<S, V>>)
                                    };

                                    if self.check_dups {
                                        shard.radix_sort_builder().sort();
                                        if shard.par_windows(2).any(|w| w[0].sig == w[1].sig) {
                                            let _ = err_send.send(SolveError::DuplicateSignature);
                                            return;
                                        }
                                    }

                                    // The second conjunct is always true on 32-bit platforms
                                    #[allow(clippy::absurd_extreme_comparisons)]
                                    if TypeId::of::<E::LocalSig>() != TypeId::of::<S>()
                                        && self.num_keys > Self::MAX_NO_LOCAL_SIG_CHECK
                                    {
                                        // Check for duplicate local signatures

                                        // E::SortSig provides a transmutable
                                        // view of SigVal with an implementation
                                        // of RadixKey that is compatible with
                                        // the sort induced by the key returned
                                        // by ShardEdge::sort_key, and equality
                                        // that implies equality of local
                                        // signatures.

                                        // SAFETY: The Arc has refcount 1 at this point
                                        // (this thread is the sole owner after receive),
                                        // so the mutable reference does not violate
                                        // aliasing. The memory layout of SigVal<S, V>
                                        // and E::SortSigVal<V> is guaranteed compatible
                                        // by the ShardEdge trait.
                                        let shard = unsafe {
                                            transmute::<
                                                &mut Vec<SigVal<S, V>>,
                                                &mut Vec<E::SortSigVal<V>>,
                                            >(shard)
                                        };

                                        let builder = shard.radix_sort_builder();
                                        if self.max_num_threads == 1 {
                                            builder
                                                .with_single_threaded_tuner()
                                                .with_parallel(false)
                                        } else {
                                            builder
                                        }
                                        .sort();

                                        let shard_len = shard.len();
                                        shard.dedup();

                                        if TypeId::of::<V>() == TypeId::of::<EmptyVal>() {
                                            // Duplicate local signatures on
                                            // large filters can be simply
                                            // removed: they do not change the
                                            // semantics of the filter because
                                            // hashes are computed on
                                            // local signatures.
                                            pl.info(format_args!(
                                                "Removed signatures: {}",
                                                shard_len - shard.len()
                                            ));
                                        } else {
                                            // For function, we have to try again
                                            if shard_len != shard.len() {
                                                let _ = err_send
                                                    .send(SolveError::DuplicateLocalSignature);
                                                return;
                                            }
                                        }
                                    } else if self.shard_edge.num_sort_keys() != 1 {
                                        // Sorting the signatures increases locality
                                        self.count_sort::<V>(shard);
                                    }
                                }

                                pl.done_with_count(shard.len());

                                main_pl.debug(format_args!(
                                    "Solving shard {}/{}...",
                                    shard_index + 1,
                                    num_shards
                                ));

                                if self.failed.load(Ordering::Relaxed) {
                                    return;
                                }

                                if TypeId::of::<V>() == TypeId::of::<EmptyVal>() {
                                    // For filters, we fill the array with random data, otherwise
                                    // elements with signature 0 would have a significantly higher
                                    // probability of being false positives.
                                    //
                                    // SAFETY: We work around the fact that [usize] does not implement Fill
                                    Mwc192::seed_from_u64(self.seed).fill_bytes(unsafe {
                                        data.as_mut_slice().align_to_mut::<u8>().1
                                    });
                                }

                                if solve_shard(self, shard_index, shard, data, &mut pl).is_err() {
                                    let _ = err_send.send(SolveError::UnsolvableShard);
                                    return;
                                }

                                if self.failed.load(Ordering::Relaxed) {
                                    return;
                                }

                                main_pl.debug(format_args!(
                                    "Completed shard {}/{}",
                                    shard_index + 1,
                                    num_shards
                                ));
                                main_pl.update();
                            }
                        }
                    }
                });
            }

            drop(err_send);
            drop(data_recv);

            if let Some(error) = err_recv.into_iter().next() {
                self.failed.store(true, Ordering::Relaxed);
                return Err(error);
            }

            Ok(())
        });

        main_pl.done();
        result
    }

    /// Peels a shard via edge indices.
    ///
    /// This peeler uses a [`SigVal`] per key (the shard), a
    /// [`ShardEdge::Vertex`] and a byte per vertex (for the [`XorGraph`]), a
    /// [`ShardEdge::Vertex`] per vertex (for the [`DoubleStack`]), and a final
    /// byte per vertex (for the stack of sides).
    ///
    /// This peeler uses more memory than [`peel_by_sig_vals_low_mem`] but
    /// less memory than [`peel_by_sig_vals_high_mem`]. It is fairly slow as
    /// it has to go through a cache-unfriendly memory indirection every time
    /// it has to retrieve a [`SigVal`] from the shard, but it is the peeler
    /// of choice when [lazy Gaussian elimination] is required, as after a
    /// failed peel-by-sig-vals it is not possible to retrieve information
    /// about the signature/value pairs in the shard.
    ///
    /// In theory one could avoid the stack of sides by putting vertices,
    /// instead of edge indices, on the upper stack, and retrieving edge
    /// indices and sides from the [`XorGraph`], as
    /// [`peel_by_sig_vals_low_mem`] does, but this would be less cache
    /// friendly. This peeler is only used for very small instances, and
    /// since we are going to pass through lazy Gaussian elimination some
    /// additional speed is a good idea.
    ///
    /// [`peel_by_sig_vals_low_mem`]: VBuilder::peel_by_sig_vals_low_mem
    /// [`peel_by_sig_vals_high_mem`]: VBuilder::peel_by_sig_vals_high_mem
    /// [lazy Gaussian elimination]: https://doi.org/10.1016/j.ic.2020.104517
    fn peel_by_index<
        'a,
        V: BinSafe,
        G: Fn(&E, SigVal<E::LocalSig, V>) -> D::Value + Send + Sync,
        H: Fn(&SigVal<S, V>) + Send + Sync,
    >(
        &self,
        shard_index: usize,
        shard: Arc<Vec<SigVal<S, V>>>,
        data: ShardData<'a, D>,
        get_val: &G,
        inspect: &H,
        pl: &mut impl ProgressLog,
    ) -> Result<PeelResult<'a, D, S, E, V>, ()> {
        let shard_edge = &self.shard_edge;
        let num_vertices = shard_edge.num_vertices();
        let num_shards = shard_edge.num_shards();

        pl.start(format!(
            "Generating graph for shard {}/{}...",
            shard_index + 1,
            num_shards
        ));

        let mut xor_graph = XorGraph::<E::Vertex>::new(num_vertices);
        for (edge_index, sig_val) in shard.iter().enumerate() {
            inspect(sig_val);
            for (side, &v) in shard_edge
                .local_edge(shard_edge.local_sig(sig_val.sig))
                .iter()
                .enumerate()
            {
                xor_graph.add(v, E::Vertex::as_from(edge_index), side);
            }
        }
        pl.done_with_count(shard.len());

        assert!(
            !xor_graph.overflow,
            "Degree overflow for shard {}/{}",
            shard_index + 1,
            num_shards
        );

        if self.failed.load(Ordering::Relaxed) {
            return Err(());
        }

        // The lower stack contains vertices to be visited. The upper stack
        // contains peeled edges.
        let mut double_stack = DoubleStack::<E::Vertex>::new(num_vertices);
        let mut sides_stack = Vec::<u8>::new();

        pl.start(format!(
            "Peeling graph for shard {}/{} by edge indices...",
            shard_index + 1,
            num_shards
        ));

        // Preload all vertices of degree one in the visit stack
        for (v, degree) in xor_graph.degrees().enumerate() {
            if degree == 1 {
                double_stack.push_lower(E::Vertex::as_from(v));
            }
        }

        while let Some(v) = double_stack.pop_lower() {
            let v: usize = v.as_to();
            if xor_graph.degree(v) == 0 {
                continue;
            }
            debug_assert!(xor_graph.degree(v) == 1);
            let (edge_index, side) = xor_graph.edge_and_side(v);
            xor_graph.zero(v);
            double_stack.push_upper(edge_index);
            sides_stack.push(side as u8);
            let edge: usize = edge_index.as_to();

            let e = shard_edge.local_edge(shard_edge.local_sig(shard[edge].sig));
            remove_edge!(
                xor_graph,
                e,
                side,
                edge_index,
                double_stack,
                push_lower,
                E::Vertex::as_from
            );
        }

        pl.done();

        if shard.len() != double_stack.upper_len() {
            pl.debug(format_args!(
                "Peeling failed for shard {}/{} (peeled {} out of {} edges, {:.2}% unpeeled)",
                shard_index + 1,
                num_shards,
                double_stack.upper_len(),
                shard.len(),
                100.0 * (shard.len() - double_stack.upper_len()) as f64 / shard.len() as f64,
            ));
            return Ok(PeelResult::Partial {
                shard_index,
                shard,
                data,
                double_stack,
                sides_stack,
            });
        }

        self.assign(
            shard_index,
            data,
            double_stack
                .iter_upper()
                .map(|&edge_index| {
                    // Turn edge indices into local edge signatures
                    // and associated values
                    let edge: usize = edge_index.as_to();
                    let sig_val = &shard[edge];
                    let local_sig = shard_edge.local_sig(sig_val.sig);
                    (
                        local_sig,
                        get_val(
                            shard_edge,
                            SigVal {
                                sig: local_sig,
                                val: sig_val.val,
                            },
                        ),
                    )
                })
                .zip(sides_stack.into_iter().rev()),
            pl,
        );

        Ok(PeelResult::Complete())
    }

    /// Peels a shard via signature/value pairs using a stack of peeled
    /// signatures/value pairs.
    ///
    /// This peeler does not need the shard once the [`XorGraph`] is built, so
    /// it drops it immediately after building the graph.
    ///
    /// It uses a [`SigVal`] and a byte per vertex (for the [`XorGraph`]), a
    /// [`ShardEdge::Vertex`] per vertex (for visit stack, albeit usually the stack
    /// never contains more than a third of the vertices), and a [`SigVal`] and
    /// a byte per key (for the stack of peeled edges).
    ///
    /// This is the fastest and more memory-consuming peeler. It has however
    /// just a small advantage during assignment with respect to
    /// [`peel_by_sig_vals_low_mem`], which uses almost half the memory. It
    /// is the peeler of choice for low levels of parallelism.
    ///
    /// [`peel_by_sig_vals_low_mem`]: VBuilder::peel_by_sig_vals_low_mem
    ///
    /// This peeler cannot be used in conjunction with [lazy Gaussian
    /// elimination] as after a failed peeling it is not possible to retrieve
    /// information about the signature/value pairs in the shard.
    ///
    /// [lazy Gaussian elimination]: https://doi.org/10.1016/j.ic.2020.104517
    fn peel_by_sig_vals_high_mem<
        V: BinSafe,
        G: Fn(&E, SigVal<E::LocalSig, V>) -> D::Value + Send + Sync,
        H: Fn(&SigVal<S, V>) + Send + Sync,
    >(
        &self,
        shard_index: usize,
        shard: Arc<Vec<SigVal<S, V>>>,
        data: ShardData<'_, D>,
        get_val: &G,
        inspect: &H,
        pl: &mut impl ProgressLog,
    ) -> Result<(), ()>
    where
        SigVal<E::LocalSig, V>: BitXor + BitXorAssign + Default,
    {
        let shard_edge = &self.shard_edge;
        let num_vertices = shard_edge.num_vertices();
        let num_shards = shard_edge.num_shards();
        let shard_len = shard.len();

        pl.start(format!(
            "Generating graph for shard {}/{}...",
            shard_index + 1,
            num_shards
        ));

        let mut xor_graph = XorGraph::<SigVal<E::LocalSig, V>>::new(num_vertices);
        for &sig_val in shard.iter() {
            inspect(&sig_val);
            let local_sig = shard_edge.local_sig(sig_val.sig);
            for (side, &v) in shard_edge.local_edge(local_sig).iter().enumerate() {
                xor_graph.add(
                    v,
                    SigVal {
                        sig: local_sig,
                        val: sig_val.val,
                    },
                    side,
                );
            }
        }
        pl.done_with_count(shard.len());

        // We are using a consuming iterator over the shard store, so this
        // drop will free the memory used by the signatures
        drop(shard);

        assert!(
            !xor_graph.overflow,
            "Degree overflow for shard {}/{}",
            shard_index + 1,
            num_shards
        );

        if self.failed.load(Ordering::Relaxed) {
            return Err(());
        }

        let mut sig_vals_stack = FastStack::<SigVal<E::LocalSig, V>>::new(shard_len);
        let mut sides_stack = FastStack::<u8>::new(shard_len);
        // Experimentally this stack never grows beyond a little more than
        // num_vertices / 4
        let mut visit_stack = Vec::<E::Vertex>::with_capacity(num_vertices / 3);

        pl.start(format!(
            "Peeling graph for shard {}/{} by signatures (high-mem)...",
            shard_index + 1,
            num_shards
        ));

        // Preload all vertices of degree one in the visit stack
        for (v, degree) in xor_graph.degrees().enumerate() {
            if degree == 1 {
                visit_stack.push(E::Vertex::as_from(v));
            }
        }

        while let Some(v) = visit_stack.pop() {
            let v: usize = v.as_to();
            if xor_graph.degree(v) == 0 {
                continue;
            }
            let (sig_val, side) = xor_graph.edge_and_side(v);
            xor_graph.zero(v);
            sig_vals_stack.push(sig_val);
            sides_stack.push(side as u8);

            let e = self.shard_edge.local_edge(sig_val.sig);
            remove_edge!(xor_graph, e, side, sig_val, visit_stack, push, |v| {
                E::Vertex::as_from(v)
            });
        }

        pl.done();
        drop(xor_graph);

        if shard_len != sig_vals_stack.len() {
            pl.debug(format_args!(
                "Peeling failed for shard {}/{} (peeled {} out of {} edges, {:.2}% unpeeled)",
                shard_index + 1,
                num_shards,
                sig_vals_stack.len(),
                shard_len,
                100.0 * (shard_len - sig_vals_stack.len()) as f64 / shard_len as f64,
            ));
            return Err(());
        }

        self.assign(
            shard_index,
            data,
            sig_vals_stack
                .iter()
                .rev()
                .map(|&sig_val| (sig_val.sig, get_val(shard_edge, sig_val)))
                .zip(sides_stack.iter().copied().rev()),
            pl,
        );

        Ok(())
    }

    /// Peels a shard via signature/value pairs using a stack of vertices to
    /// represent peeled edges.
    ///
    /// This peeler does not need the shard once the [`XorGraph`] is built, so
    /// it drops it immediately after building the graph.
    ///
    /// It uses a [`SigVal`] and a byte per vertex (for the [`XorGraph`]) and a
    /// [`ShardEdge::Vertex`] per vertex (for a [`DoubleStack`]).
    ///
    /// This is by far the less memory-hungry peeler, and it is just slightly
    /// slower than [`peel_by_sig_vals_high_mem`], which uses almost twice
    /// the memory. It is the peeler of choice for significant levels of
    /// parallelism.
    ///
    /// [`peel_by_sig_vals_high_mem`]: VBuilder::peel_by_sig_vals_high_mem
    ///
    /// This peeler cannot be used in conjunction with [lazy Gaussian
    /// elimination] as after a failed peeling it is not possible to retrieve
    /// information about the signature/value pairs in the shard.
    ///
    /// [lazy Gaussian elimination]: https://doi.org/10.1016/j.ic.2020.104517
    fn peel_by_sig_vals_low_mem<
        V: BinSafe,
        G: Fn(&E, SigVal<E::LocalSig, V>) -> D::Value + Send + Sync,
        H: Fn(&SigVal<S, V>) + Send + Sync,
    >(
        &self,
        shard_index: usize,
        shard: Arc<Vec<SigVal<S, V>>>,
        data: ShardData<'_, D>,
        get_val: &G,
        inspect: &H,
        pl: &mut impl ProgressLog,
    ) -> Result<(), ()>
    where
        SigVal<E::LocalSig, V>: BitXor + BitXorAssign + Default,
    {
        let shard_edge = &self.shard_edge;
        let num_vertices = shard_edge.num_vertices();
        let num_shards = shard_edge.num_shards();
        let shard_len = shard.len();

        pl.start(format!(
            "Generating graph for shard {}/{}...",
            shard_index + 1,
            num_shards,
        ));

        let mut xor_graph = XorGraph::<SigVal<E::LocalSig, V>>::new(num_vertices);
        for &sig_val in shard.iter() {
            inspect(&sig_val);
            let local_sig = shard_edge.local_sig(sig_val.sig);
            for (side, &v) in shard_edge.local_edge(local_sig).iter().enumerate() {
                xor_graph.add(
                    v,
                    SigVal {
                        sig: local_sig,
                        val: sig_val.val,
                    },
                    side,
                );
            }
        }
        pl.done_with_count(shard.len());

        // We are using a consuming iterator over the shard store, so this
        // drop will free the memory used by the signatures
        drop(shard);

        assert!(
            !xor_graph.overflow,
            "Degree overflow for shard {}/{}",
            shard_index + 1,
            num_shards
        );

        if self.failed.load(Ordering::Relaxed) {
            return Err(());
        }

        let mut visit_stack = DoubleStack::<E::Vertex>::new(num_vertices);

        pl.start(format!(
            "Peeling graph for shard {}/{} by signatures (low-mem)...",
            shard_index + 1,
            num_shards
        ));

        // Preload all vertices of degree one in the visit stack
        for (v, degree) in xor_graph.degrees().enumerate() {
            if degree == 1 {
                visit_stack.push_lower(E::Vertex::as_from(v));
            }
        }

        while let Some(v) = visit_stack.pop_lower() {
            let v: usize = v.as_to();
            if xor_graph.degree(v) == 0 {
                continue;
            }
            let (sig_val, side) = xor_graph.edge_and_side(v);
            xor_graph.zero(v);
            visit_stack.push_upper(E::Vertex::as_from(v));

            let e = self.shard_edge.local_edge(sig_val.sig);
            remove_edge!(xor_graph, e, side, sig_val, visit_stack, push_lower, |v| {
                E::Vertex::as_from(v)
            });
        }

        pl.done();

        if shard_len != visit_stack.upper_len() {
            pl.debug(format_args!(
                "Peeling failed for shard {}/{} (peeled {} out of {} edges, {:.2}% unpeeled)",
                shard_index + 1,
                num_shards,
                visit_stack.upper_len(),
                shard_len,
                100.0 * (shard_len - visit_stack.upper_len()) as f64 / shard_len as f64,
            ));
            return Err(());
        }

        self.assign(
            shard_index,
            data,
            visit_stack.iter_upper().map(|&v| {
                let (sig_val, side) = xor_graph.edge_and_side(v.as_to());
                ((sig_val.sig, get_val(shard_edge, sig_val)), side as u8)
            }),
            pl,
        );

        Ok(())
    }

    /// Solves a shard of given index possibly using [lazy Gaussian
    /// elimination], and stores the solution in the given data.
    ///
    /// As a first try, the shard is [peeled by index]. If the peeling is
    /// [partial], lazy Gaussian elimination is used to solve the remaining
    /// edges.
    ///
    /// [peeled by index]: VBuilder::peel_by_index
    /// [partial]: PeelResult::Partial
    ///
    /// This method will scan the double stack, without emptying it, to check
    /// which edges have been peeled. The information will be then passed to
    /// [`VBuilder::assign`] to complete the assignment of values.
    ///
    /// [lazy Gaussian elimination]: https://doi.org/10.1016/j.ic.2020.104517
    fn lge_shard<
        V: BinSafe,
        G: Fn(&E, SigVal<E::LocalSig, V>) -> D::Value + Send + Sync,
        H: Fn(&SigVal<S, V>) + Send + Sync,
    >(
        &self,
        shard_index: usize,
        shard: Arc<Vec<SigVal<S, V>>>,
        data: ShardData<'_, D>,
        get_val: &G,
        inspect: &H,
        pl: &mut impl ProgressLog,
    ) -> Result<(), ()> {
        let shard_edge = &self.shard_edge;
        // Let's try to peel first
        match self.peel_by_index(shard_index, shard, data, get_val, inspect, pl) {
            Err(()) => Err(()),
            Ok(PeelResult::Complete()) => Ok(()),
            Ok(PeelResult::Partial {
                shard_index,
                shard,
                mut data,
                double_stack,
                sides_stack,
            }) => {
                pl.debug(format_args!("Switching to lazy Gaussian elimination..."));
                // We now solve the remaining edges with Gaussian elimination.
                pl.start(format!(
                    "Generating system for shard {}/{}...",
                    shard_index + 1,
                    shard_edge.num_shards()
                ));

                let num_vertices = shard_edge.num_vertices();
                let mut peeled_edges: BitVec = BitVec::new(shard.len());
                let mut used_vars: BitVec = BitVec::new(num_vertices);
                for &edge in double_stack.iter_upper() {
                    peeled_edges.set(edge.as_to(), true);
                }

                // Create data for a system using non-peeled edges
                //
                // SAFETY: there is no undefined behavior here, but the
                // raw construction methods we use assume that the
                // equations are sorted, that the variables are not repeated,
                // and the variables are in the range [0..num_vertices)
                let mut system = unsafe {
                    crate::utils::mod2_sys::Modulo2System::from_parts(
                        num_vertices,
                        shard
                            .iter()
                            .enumerate()
                            .filter(|(edge_index, _)| !peeled_edges[*edge_index])
                            .map(|(_edge_index, sig_val)| {
                                let local_sig = shard_edge.local_sig(sig_val.sig);
                                let mut eq: Vec<_> = shard_edge
                                    .local_edge(local_sig)
                                    .iter()
                                    .map(|&x| {
                                        used_vars.set(x, true);
                                        x as u32
                                    })
                                    .collect();
                                eq.sort_unstable();
                                crate::utils::mod2_sys::Modulo2Equation::from_parts(
                                    eq,
                                    get_val(
                                        shard_edge,
                                        SigVal {
                                            sig: local_sig,
                                            val: sig_val.val,
                                        },
                                    ),
                                )
                            })
                            .collect(),
                    )
                };

                if self.failed.load(Ordering::Relaxed) {
                    return Err(());
                }

                pl.start("Solving system...");
                let result = system.lazy_gaussian_elimination().map_err(|_| ())?;
                pl.done_with_count(system.num_equations());

                for (v, &value) in result.iter().enumerate().filter(|(v, _)| used_vars[*v]) {
                    data.set_value(v, value);
                }

                self.assign(
                    shard_index,
                    data,
                    double_stack
                        .iter_upper()
                        .map(|&edge_index| {
                            let edge: usize = edge_index.as_to();
                            let sig_val = &shard[edge];
                            let local_sig = shard_edge.local_sig(sig_val.sig);
                            (
                                local_sig,
                                get_val(
                                    shard_edge,
                                    SigVal {
                                        sig: local_sig,
                                        val: sig_val.val,
                                    },
                                ),
                            )
                        })
                        .zip(sides_stack.into_iter().rev()),
                    pl,
                );
                Ok(())
            }
        }
    }

    /// Perform assignment of values based on peeling data.
    ///
    /// This method might be called after a successful peeling procedure, or
    /// after a linear solver has been used to solve the remaining edges.
    ///
    /// `sigs_vals_sides` is an iterator returning pairs of signature/value pairs
    /// and sides in reverse peeling order.
    fn assign(
        &self,
        shard_index: usize,
        mut data: ShardData<'_, D>,
        sigs_vals_sides: impl Iterator<Item = ((E::LocalSig, D::Value), u8)>,
        pl: &mut impl ProgressLog,
    ) where
        for<'a> ShardData<'a, D>: SliceByValueMut,
    {
        if self.failed.load(Ordering::Relaxed) {
            return;
        }

        pl.start(format!(
            "Assigning values for shard {}/{}...",
            shard_index + 1,
            self.shard_edge.num_shards()
        ));

        for ((sig, val), side) in sigs_vals_sides {
            let edge = self.shard_edge.local_edge(sig);
            let side = side as usize;
            // SAFETY: vertex indices from `local_edge` are guaranteed within
            // bounds of `data` by the ShardEdge contract; `side` is always
            // 0, 1, or 2 because it encodes a hyperedge vertex index.
            unsafe {
                let xor = match side {
                    0 => data.get_value_unchecked(edge[1]) ^ data.get_value_unchecked(edge[2]),
                    1 => data.get_value_unchecked(edge[0]) ^ data.get_value_unchecked(edge[2]),
                    2 => data.get_value_unchecked(edge[0]) ^ data.get_value_unchecked(edge[1]),
                    _ => core::hint::unreachable_unchecked(),
                };

                data.set_value_unchecked(edge[side], val ^ xor);
            }
        }
        pl.done();
    }
}
