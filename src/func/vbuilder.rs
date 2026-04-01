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
use log::info;
use num_primitive::PrimitiveNumber;
use rand::rngs::SmallRng;
use rand::{Rng, RngExt, SeedableRng};
use rayon::iter::ParallelIterator;
use rayon::slice::ParallelSlice;
use rdst::*;
use std::any::TypeId;
use std::borrow::{Borrow, Cow};
use std::hint::unreachable_unchecked;
use std::marker::PhantomData;
use std::mem::transmute;
use std::ops::{BitXor, BitXorAssign};
use std::slice::Iter;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;
use thread_priority::ThreadPriority;
use value_traits::slices::{SliceByValue, SliceByValueMut};

use super::shard_edge::FuseLge3Shards;

const LOG2_MAX_SHARDS: u32 = 16;

/// A builder for [`VFunc`] and [`VFilter`].
///
/// Keys must implement the [`ToSig`] trait, which provides a method to compute
/// a signature of the key.
///
/// There are two construction modes: in core memory (default) and
/// [offline](VBuilder::offline); both use a [`SigStore`]. In the first case,
/// space will be allocated in core memory for signatures and associated values
/// for all keys; in the second case, such information will be stored in a
/// number of on-disk buckets.
///
/// There are several setters: for example, you can [set the maximum number
/// of threads](VBuilder::max_num_threads).
///
/// Once signatures have been computed, each parallel thread will process a
/// shard of the signature/value pairs. For very large key sets shards will be
/// significantly smaller than the number of keys, so the memory usage, in
/// particular in offline mode, can be significantly reduced. Note that using
/// too many threads might actually be harmful due to memory contention.
///
/// The generic parameters are explained in the [`VFunc`] documentation. You
/// have to choose the type of the output values and the backend. The remaining
/// parameters have default values that are the same as those of
/// [`VFunc`]/[`VFilter`], and some elaboration about them can be found in their
/// documentation.
///
/// All construction methods require to pass one or two [`FallibleRewindableLender`]s
/// (keys and possibly values), and the construction might fail and keys might
/// be scanned again. The structures in the [`lenders`] module provide easy ways
/// to build such lenders, even starting from compressed files of UTF-8 strings.
/// The type of the keys of the resulting filter or function will be the type of
/// the elements of the [`FallibleRewindableLender`].
///
/// # Signed index functions
///
/// The methods [`try_build_sig_index`](VBuilder::try_build_sig_index) and
/// [`try_build_bit_sig_index`](VBuilder::try_build_bit_sig_index) build index
/// functions (i.e., functions mapping elements of a list to their rank) and
/// they associate hashes to keys, so the result of the index function can be
/// checked. See the documentation of the
/// [`signed_vfunc`](crate::dict::signed_vfunc) module for more details.
///
/// # Building from stores
///
/// The low-level method
/// [`try_build_func_with_store`](VBuilder::try_build_func_with_store)
/// builds a function from a [`ShardStore`] containing signatures and
/// values. The store is expected to be already populated; the method
/// recalculates sharding for the actual key count. It can be combined
/// with [`FilteredShardStore`] to build from a filtered subset. This
/// is used when building compound functions such as [`VFunc2`].
///
/// See [`VFunc::try_new`], [`VFilter::try_new`], etc. for usage examples.

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

    /// The maximum number of parallel threads to use. The default is 8.
    #[setters(generate = true)]
    #[derivative(Default(value = "8"))]
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
    low_mem: Option<bool>,

    /// The seed for the random number generator.
    #[setters(generate = true)]
    seed: u64,

    /// The base-2 logarithm of buckets of the [`SigStore`]. The default is 8.
    /// This value is automatically overridden, even if set, if you provide an
    /// [expected number of keys](VBuilder::expected_num_keys).
    #[setters(generate = true, strip_option)]
    #[derivative(Default(value = "8"))]
    log2_buckets: u32,

    /// The target relative space loss due to [ε-cost
    /// sharding](https://doi.org/10.4230/LIPIcs.ESA.2019.38).
    ///
    /// The default is 0.001. Setting a larger target, for example, 0.01, will
    /// increase the space overhead due to sharding, but will provide in general
    /// finer shards. This might not always happen, however, because the ε-cost
    /// bound is just one of the bounds concurring in limiting the amount of
    /// sharding for a specific [`ShardEdge`]. For example, increasing the
    /// target to 0.01 will provide very fine sharding using `Mwhc3Shards`
    /// shard/edge logic, activated by the `mwhc` feature, but will affect only
    /// slightly [`FuseLge3Shards`] or
    /// [`FuseLge3FullSigs`](crate::func::shard_edge::FuseLge3FullSigs).
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
    c: f64,
    /// Whether we should use [lazy Gaussian
    /// elimination](https://doi.org/10.1016/j.ic.2020.104517).
    lge: bool,
    /// The number of threads to use.
    num_threads: usize,
    /// Fast-stop for failed attempts.
    failed: AtomicBool,
    #[doc(hidden)]
    _marker_v: PhantomData<(D, S)>,
}

impl<D: BitFieldSlice<Value: Word + BinSafe> + Send + Sync, S: Sig, E: ShardEdge<S, 3>>
    VBuilder<D, S, E>
{
    /// Sets up shards from the expected number of keys and returns the
    /// seed. This is the same initialization that
    /// [`try_populate_and_build`](Self::try_populate_and_build)
    /// performs at the start; callers that drive `try_solve_once`
    /// directly must call this first.
    pub(crate) fn init_shards_and_seed(&mut self) -> u64 {
        if let Some(expected_num_keys) = self.expected_num_keys {
            self.shard_edge.set_up_shards(expected_num_keys, self.eps);
            self.log2_buckets = self.shard_edge.shard_high_bits();
        }
        self.seed
    }
}

/// Fatal build errors.
#[derive(thiserror::Error, Debug)]
pub enum BuildError {
    /// A duplicate key was detected.
    #[error("Duplicate key")]
    DuplicateKey,
    /// Duplicate local signatures were detected.
    #[error("Duplicate local signatures: use full signatures")]
    DuplicateLocalSignatures,
    /// A value is too large for the specified bit size.
    #[error("Value too large for specified bit size")]
    ValueTooLarge,
}

/// Transient error during the build, leading to
/// trying with a different seed.
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
    W: Word + BinSafe + Send + Sync,
    D: BitFieldSlice<Value = W> + BitFieldSliceMut<Value = W> + Send + Sync + 'a,
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
        _marker: PhantomData<W>,
    },
}

/// A graph represented compactly.
///
/// Each (*k*-hyper)edge is a set of *k* vertices (by construction fuse graphs
/// do not have degenerate edges), but we represent it internally as a vector.
/// We call *side* the position of a vertex in the edge.
///
/// For each vertex we store information about the edges incident to the vertex
/// and the sides of the vertex in such edges. While technically not necessary
/// to perform peeling, the knowledge of the sides speeds up the peeling visit
/// by reducing the number of tests that are necessary to update the degrees
/// once the edge is peeled (see the `peel_by_*` methods). For the same reason
/// it also speeds up assignment.
///
/// Depending on the peeling method (by signature or by index), the graph will
/// store edge indices or signature/value pairs (the generic parameter `X`).
///
/// Edge information is packed together using Djamal's XOR trick (see
/// [“Cache-Oblivious Peeling of Random
/// Hypergraphs”](https://doi.org/10.1109/DCC.2014.48)): since during the
/// peeling visit we need to know the content of the list only when a single
/// edge index is present, we can XOR together all the edge information.
///
/// We use a single byte to store the degree (six upper bits) and the XOR of the
/// sides (lower two bits). The degree can be stored with a small number of bits
/// because the graph is random, so the maximum degree is *O*(log log *n*).
/// In any case, the Boolean field `overflow` will become `true` in case of
/// overflow.
///
/// When we peel an edge, we just [zero the degree](Self::zero), leaving the
/// edge information and the side in place for further processing later.
struct XorGraph<X: Copy + Default + BitXor + BitXorAssign> {
    edges: Box<[X]>,
    degrees_sides: Box<[u8]>,
    overflow: bool,
}

impl<X: BitXor + BitXorAssign + Default + Copy> XorGraph<X> {
    pub fn new(n: usize) -> XorGraph<X> {
        XorGraph {
            edges: vec![X::default(); n].into(),
            degrees_sides: vec![0; n].into(),
            overflow: false,
        }
    }

    #[inline(always)]
    pub fn add(&mut self, v: usize, x: X, side: usize) {
        debug_assert!(side < 3);
        let (degree_size, overflow) = self.degrees_sides[v].overflowing_add(4);
        self.degrees_sides[v] = degree_size;
        self.overflow |= overflow;
        self.degrees_sides[v] ^= side as u8;
        self.edges[v] ^= x;
    }

    #[inline(always)]
    pub fn remove(&mut self, v: usize, x: X, side: usize) {
        debug_assert!(side < 3);
        self.degrees_sides[v] -= 4;
        self.degrees_sides[v] ^= side as u8;
        self.edges[v] ^= x;
    }

    #[inline(always)]
    pub fn zero(&mut self, v: usize) {
        self.degrees_sides[v] &= 0b11;
    }

    #[inline(always)]
    pub fn edge_and_side(&self, v: usize) -> (X, usize) {
        debug_assert!(self.degree(v) < 2);
        (self.edges[v] as _, (self.degrees_sides[v] & 0b11) as _)
    }

    #[inline(always)]
    pub fn degree(&self, v: usize) -> u8 {
        self.degrees_sides[v] >> 2
    }

    pub fn degrees(
        &self,
    ) -> std::iter::Map<std::iter::Copied<std::slice::Iter<'_, u8>>, fn(u8) -> u8> {
        self.degrees_sides.iter().copied().map(|d| d >> 2)
    }
}

/// A preallocated stack implementation that avoids the expensive (even if
/// rarely taken) branch of the `Vec` implementation in which memory is
/// reallocated. Note that using [`Vec::with_capacity`] is not enough, because
/// for the CPU the branch is still there.
struct FastStack<X: Copy + Default> {
    stack: Vec<X>,
    top: usize,
}

impl<X: Copy + Default> FastStack<X> {
    pub fn new(n: usize) -> FastStack<X> {
        FastStack {
            stack: vec![X::default(); n],
            top: 0,
        }
    }

    pub fn push(&mut self, x: X) {
        debug_assert!(self.top < self.stack.len());
        self.stack[self.top] = x;
        self.top += 1;
    }

    pub fn len(&self) -> usize {
        self.top
    }

    pub fn iter(&self) -> std::slice::Iter<'_, X> {
        self.stack[..self.top].iter()
    }
}

/// Two stacks in the same vector.
///
/// This struct implements a pair of stacks sharing the same memory. The lower
/// stack grows from the beginning of the vector, the upper stack grows from the
/// end of the vector. Since we use the lower stack for the visit and the upper
/// stack for peeled edges (possibly represented by the vertex from which they
/// have been peeled), the sum of the lengths of the two stacks cannot exceed
/// the length of the vector.
#[derive(Debug)]
struct DoubleStack<V> {
    stack: Vec<V>,
    lower: usize,
    upper: usize,
}

impl<V: Default + Copy> DoubleStack<V> {
    fn new(n: usize) -> DoubleStack<V> {
        DoubleStack {
            stack: vec![V::default(); n],
            lower: 0,
            upper: n,
        }
    }
}

impl<V: Copy> DoubleStack<V> {
    #[inline(always)]
    fn push_lower(&mut self, v: V) {
        debug_assert!(self.lower < self.upper);
        self.stack[self.lower] = v;
        self.lower += 1;
    }

    #[inline(always)]
    fn push_upper(&mut self, v: V) {
        debug_assert!(self.lower < self.upper);
        self.upper -= 1;
        self.stack[self.upper] = v;
    }

    #[inline(always)]
    fn pop_lower(&mut self) -> Option<V> {
        if self.lower == 0 {
            None
        } else {
            self.lower -= 1;
            Some(self.stack[self.lower])
        }
    }

    fn upper_len(&self) -> usize {
        self.stack.len() - self.upper
    }

    fn iter_upper(&self) -> Iter<'_, V> {
        self.stack[self.upper..].iter()
    }
}

/// An iterator over segments of data associated with each shard.
type ShardDataIter<'a, D> = <D as SliceByValueMut>::ChunksMut<'a>;
/// A segment of data associated with a specific shard.
type ShardData<'a, D> = <ShardDataIter<'a, D> as Iterator>::Item;

impl<W: Word + BinSafe + AsU128, S: Sig + Send + Sync, E: ShardEdge<S, 3>>
    VBuilder<BitFieldVec<Box<[W]>>, S, E>
where
    SigVal<S, W>: RadixKey,
    SigVal<E::LocalSig, W>: BitXor + BitXorAssign,
{
    /// Builds a new function by reusing an existing [`ShardStore`] with
    /// remapped values.
    ///
    /// This is a low-level method that requires a thorough understanding
    /// of the builder's internal state.
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
    ///
    /// # Arguments
    ///
    /// * `seed` — the seed from the store's population step.
    ///
    /// * `shard_edge` — the shard edge from the same population step.
    ///
    /// * `max_value` — the maximum value that `get_val` can return
    ///   (determines the bit width of the output).
    ///
    /// * `shard_store` — the store kept from a previous
    ///   `try_build_func_and_store(…, drain_store = false, …)` call.
    ///
    /// * `get_val` — maps each [`SigVal<E::LocalSig, V>`](SigVal) to
    ///   the new output value `W`.
    ///
    /// * `inspect` — a closure that is called on each signature/value pair
    ///   during the build, for example, to log the distribution of values.
    ///
    /// * `pl` — the progress logger.
    pub fn try_build_func_with_store<T: ?Sized + ToSig<S>, V: BinSafe + Default + Send + Sync>(
        mut self,
        seed: u64,
        shard_edge: E,
        max_value: W,
        shard_store: &mut (impl ShardStore<S, V> + ?Sized),
        get_val: &(impl Fn(&E, SigVal<E::LocalSig, V>) -> W + Send + Sync),
        inspect: &(impl Fn(&SigVal<S, V>) + Send + Sync),
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> anyhow::Result<VFunc<T, BitFieldVec<Box<[W]>>, S, E>>
    where
        SigVal<S, V>: RadixKey,
        SigVal<E::LocalSig, V>: BitXor + BitXorAssign,
        for<'a> ShardDataIter<'a, BitFieldVec<Box<[W]>>>: Send,
        for<'a> <ShardDataIter<'a, BitFieldVec<Box<[W]>>> as Iterator>::Item: Send,
    {
        self.shard_edge = shard_edge;
        self.num_keys = shard_store.len();
        self.bit_width = max_value.as_u128().bit_len() as usize;

        // The shard structure is fixed by the store (set at
        // `into_shard_store` time). We only reconfigure graph
        // parameters (c, lge, segment size, vertex count) for the
        // actual key count and shard sizes.
        let max_shard = shard_store.shard_sizes().iter().copied().max().unwrap_or(0);
        (self.c, self.lge) = self.shard_edge.set_up_graphs(self.num_keys, max_shard);

        pl.info(format_args!(
            "Number of keys: {} Max value: {} Bit width: {}",
            self.num_keys,
            {
                let v: u128 = max_value.as_u128();
                v
            },
            self.bit_width,
        ));

        let data: BitFieldVec<Box<[W]>> = BitFieldVec::<Box<[W]>>::new_unaligned(
            self.bit_width,
            self.shard_edge.num_vertices() * self.shard_edge.num_shards(),
        );

        self.try_build_from_shard_iter(seed, data, shard_store.iter(), get_val, inspect, pl)
            .map_err(Into::into)
    }
}

/// Handles the result of [`VBuilder::try_solve_once`], returning on
/// success or non-retryable error and falling through on retryable
/// [`SolveError`] variants so the caller's loop can retry with a
/// different seed.
macro_rules! handle_solve_result {
    ($result:expr, $dup_count:expr, $local_dup_count:expr, $pl:expr) => {
        match $result {
            Ok(r) => return Ok(r),
            Err(error) => match error.downcast::<SolveError>() {
                Ok(vfunc_error) => match vfunc_error {
                    SolveError::DuplicateSignature => {
                        if $dup_count >= 3 {
                            $pl.error(format_args!("Duplicate keys (duplicate 128-bit signatures with four different seeds)"));
                            return Err(BuildError::DuplicateKey.into());
                        }
                        $pl.warn(format_args!(
                            "Duplicate 128-bit signature, trying again with a different seed..."
                        ));
                        $dup_count += 1;
                    }
                    SolveError::DuplicateLocalSignature => {
                        if $local_dup_count >= 2 {
                            $pl.error(format_args!("Duplicate local signatures: use full signatures (duplicate local signatures with three different seeds)"));
                            return Err(BuildError::DuplicateLocalSignatures.into());
                        }
                        $pl.warn(format_args!(
                            "Duplicate local signature, trying again with a different seed..."
                        ));
                        $local_dup_count += 1;
                    }
                    SolveError::MaxShardTooBig => {
                        $pl.warn(format_args!(
                            "The maximum shard is too big, trying again with a different seed..."
                        ));
                    }
                    SolveError::UnsolvableShard => {
                        $pl.warn(format_args!(
                            "Unsolvable shard, trying again with a different seed..."
                        ));
                    }
                },
                Err(error) => return Err(error),
            },
        }
    };
}

impl<
    D: BitFieldSlice<Value: Word + BinSafe + AsU128> + Send + Sync,
    S: Sig + Send + Sync,
    E: ShardEdge<S, 3>,
> VBuilder<D, S, E>
{
    /// Builds a new [`VFunc`], optionally retaining the populated shard
    /// store for reuse.
    ///
    /// This is a low-level method that requires a thorough understanding
    /// of the builder's internal state.
    ///
    /// `new_data(bit_width, len)` allocates the backend storage of the
    /// given bit width and length.
    ///
    /// The second element of the returned tuple is the store that was
    /// populated during construction. The caller can pass it to
    /// [`try_build_func_with_store`](Self::try_build_func_with_store)
    /// to build additional functions without re-hashing the keys, or
    /// simply drop it.
    ///
    /// If `drain_store` is `true`, the store is drained during
    /// construction (freeing memory as shards are consumed); the
    /// returned store is empty. If `false`, the store is preserved
    /// intact for reuse.
    pub fn try_build_func_and_store<
        T: ?Sized + ToSig<S> + std::fmt::Debug,
        B: ?Sized + Borrow<T>,
        P: ProgressLog + Clone + Send + Sync,
    >(
        mut self,
        keys: impl FallibleRewindableLender<
            RewindError: Error + Send + Sync + 'static,
            Error: Error + Send + Sync + 'static,
        > + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
        values: impl FallibleRewindableLender<
            RewindError: Error + Send + Sync + 'static,
            Error: Error + Send + Sync + 'static,
        > + for<'lend> FallibleLending<'lend, Lend = &'lend D::Value>,
        new_data: fn(usize, usize) -> D,
        drain_store: bool,
        pl: &mut P,
    ) -> anyhow::Result<(
        VFunc<T, D, S, E>,
        Box<dyn ShardStore<S, D::Value> + Send + Sync>,
    )>
    where
        D::Value: AsU128,
        SigVal<S, D::Value>: RadixKey,
        SigVal<E::LocalSig, D::Value>: BitXor + BitXorAssign,
        D: for<'a> BitFieldSliceMut<ChunksMut<'a>: Iterator<Item: BitFieldSliceMut>>,
        for<'a> ShardDataIter<'a, D>: Send,
        for<'a> <ShardDataIter<'a, D> as Iterator>::Item: Send,
    {
        let get_val = |_shard_edge: &E, sig_val: SigVal<E::LocalSig, D::Value>| sig_val.val;

        self.try_populate_and_build(
            keys,
            values,
            &mut |builder, seed, mut store, max_value, _num_keys, pl: &mut P, _state: &mut ()| {
                builder.bit_width = max_value.as_u128().bit_len() as usize;

                let data = new_data(
                    builder.bit_width,
                    builder.shard_edge.num_vertices() * builder.shard_edge.num_shards(),
                );

                pl.info(format_args!(
                    "Number of keys: {} Max value: {} Bit width: {}",
                    builder.num_keys,
                    {
                        let v: u128 = max_value.as_u128();
                        v
                    },
                    builder.bit_width,
                ));

                let shard_iter: Box<dyn Iterator<Item = _> + Send + Sync + '_> = if drain_store {
                    store.drain()
                } else {
                    store.iter()
                };
                let func = builder.try_build_from_shard_iter(
                    seed,
                    data,
                    shard_iter,
                    &get_val,
                    &|_| {},
                    pl,
                )?;
                Ok((func, store))
            },
            pl,
            (),
        )
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
    /// `build_fn` is called with `(&mut self, seed, store, max_value,
    /// num_keys, pl)`. The builder's `shard_edge`, `c`, and `lge` fields
    /// are already set up when `build_fn` is invoked, so it can call
    /// `try_build_from_shard_iter`
    /// directly.
    ///
    /// Returns whatever `build_fn` returns on success.
    pub fn try_populate_and_build<
        T: ?Sized + ToSig<S> + std::fmt::Debug,
        B: ?Sized + Borrow<T>,
        V: BinSafe + Default + Send + Sync + Ord + AsU128,
        R,
        P: ProgressLog + Clone + Send + Sync,
        C,
    >(
        &mut self,
        mut keys: impl FallibleRewindableLender<
            RewindError: Error + Send + Sync + 'static,
            Error: Error + Send + Sync + 'static,
        > + for<'lend> FallibleLending<'lend, Lend = &'lend B>,
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
    ) -> anyhow::Result<R>
    where
        SigVal<S, V>: RadixKey,
    {
        if let Some(expected_num_keys) = self.expected_num_keys {
            self.shard_edge.set_up_shards(expected_num_keys, self.eps);
            self.log2_buckets = self.shard_edge.shard_high_bits();
        }

        let mut dup_count = 0u32;
        let mut local_dup_count = 0u32;
        let mut prng = SmallRng::seed_from_u64(self.seed);

        pl.info(format_args!("Using 2^{} buckets", self.log2_buckets));

        loop {
            let seed: u64 = prng.random();

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
                            sig: T::to_sig(key.borrow(), seed),
                            val: maybe_val,
                        })?;
                    }
                    Ok(maybe_max_value)
                };

                self.try_solve_once(seed, &mut populate, build_fn, pl, &mut state)
            };

            handle_solve_result!(result, dup_count, local_dup_count, pl);

            values = values.rewind()?;
            keys = keys.rewind()?;
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
        V: BinSafe + Default + Send + Sync + Ord + AsU128,
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

    /// Inner generic implementation of [`try_solve_once`](Self::try_solve_once).
    ///
    /// This is generic over `SS` so that the `populate` closure can push
    /// to the concrete store type without dynamic dispatch. The `state`
    /// parameter is forwarded to both `populate` and `build_fn`.
    fn try_solve_once_inner<
        V: BinSafe + Default + Send + Sync + Ord + AsU128,
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
            "Computing and storing {}-bit signatures in {} using seed 0x{:016x}...",
            std::mem::size_of::<S>() * 8,
            sig_store
                .temp_dir()
                .map(|d| d.path().to_string_lossy())
                .unwrap_or(Cow::Borrowed("memory")),
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

        info!(
            "Computation of signatures from inputs completed in {:.3} seconds ({} keys, {:.3} ns/key)",
            start.elapsed().as_secs_f64(),
            num_keys,
            start.elapsed().as_nanos() as f64 / num_keys as f64
        );

        let shard_edge = &mut self.shard_edge;
        shard_edge.set_up_shards(num_keys, self.eps);

        let start = Instant::now();

        let shard_store = sig_store.into_shard_store(shard_edge.shard_high_bits())?;
        let max_shard = shard_store.shard_sizes().iter().copied().max().unwrap_or(0);

        if shard_edge.shard_high_bits() != 0 {
            pl.info(format_args!(
                "Max shard / average shard: {:.2}%",
                (100.0 * max_shard as f64) / (num_keys as f64 / shard_edge.num_shards() as f64)
            ));
        }

        if max_shard as f64 > 1.01 * num_keys as f64 / shard_edge.num_shards() as f64 {
            return Err(SolveError::MaxShardTooBig.into());
        }

        (self.c, self.lge) = shard_edge.set_up_graphs(num_keys, max_shard);
        self.num_keys = num_keys;

        let store = Box::new(shard_store) as Box<dyn ShardStore<S, V> + Send + Sync>;

        build_fn(self, seed, store, maybe_max_value, num_keys, pl, state).inspect(|_| {
            info!(
                "Construction from signatures completed in {:.3} seconds ({} keys, {:.3} ns/key)",
                start.elapsed().as_secs_f64(),
                num_keys,
                start.elapsed().as_nanos() as f64 / num_keys as f64
            );
        })
    }
}

impl<
    D: BitFieldSlice<Value: Word + BinSafe>
        + for<'a> BitFieldSliceMut<ChunksMut<'a>: Iterator<Item: BitFieldSliceMut>>
        + Send
        + Sync,
    S: Sig + Send + Sync,
    E: ShardEdge<S, 3>,
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
    /// * `data` must be freshly allocated, zero-initialized storage of
    ///   size `shard_edge.num_vertices() * shard_edge.num_shards()`.
    /// * `self.shard_edge`, `self.c`, `self.lge`, `self.bit_width`, and
    ///   `self.num_keys` must be set up by the caller (typically by
    ///   [`try_solve_once`](Self::try_solve_once)).
    /// * `get_val` must be deterministic.
    ///
    /// # Errors
    ///
    /// Returns [`SolveError::UnsolvableShard`] or
    /// [`SolveError::DuplicateLocalSignature`] if the system cannot be
    /// solved with the current seed.
    ///
    /// The peeling algorithm is selected based on `self.lge` and
    /// `self.low_mem`; see the [`low_mem`](VBuilder::low_mem) field
    /// documentation for the automatic selection heuristic.
    pub(crate) fn try_build_from_shard_iter<
        T: ?Sized + ToSig<S>,
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
    ) -> Result<VFunc<T, D, S, E>, SolveError>
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

        pl.info(format_args!("{}", self.shard_edge));
        pl.info(format_args!(
            "c: {}, Overhead: {:+.4}% Number of threads: {}",
            self.c,
            100. * ((shard_edge.num_vertices() * shard_edge.num_shards()) as f64
                / (self.num_keys as f64)
                - 1.),
            self.num_threads
        ));

        if self.lge {
            pl.info(format_args!(
                "Peeling with lazy Gaussian elimination fallback"
            ));
            self.par_solve(
                shard_iter,
                &mut data,
                |this, shard_index, shard, data, pl| {
                    this.lge_shard(shard_index, shard, data, get_val, inspect, pl)
                },
                &mut pl.concurrent(),
                pl,
            )?;
        } else if self.low_mem == Some(true)
            || self.low_mem.is_none() && self.num_threads > 3 && shard_edge.num_shards() > 2
        {
            // Less memory, but slower
            self.par_solve(
                shard_iter,
                &mut data,
                |this, shard_index, shard, data, pl| {
                    this.peel_by_sig_vals_low_mem(shard_index, shard, data, get_val, inspect, pl)
                },
                &mut pl.concurrent(),
                pl,
            )?;
        } else {
            // More memory, but faster
            self.par_solve(
                shard_iter,
                &mut data,
                |this, shard_index, shard, data, pl| {
                    this.peel_by_sig_vals_high_mem(shard_index, shard, data, get_val, inspect, pl)
                },
                &mut pl.concurrent(),
                pl,
            )?;
        }

        pl.info(format_args!(
            "Bits/keys: {} ({:+.4}%)",
            data.len() as f64 * self.bit_width as f64 / self.num_keys as f64,
            100.0 * (data.len() as f64 / self.num_keys as f64 - 1.),
        ));

        Ok(VFunc {
            seed,
            shard_edge: self.shard_edge,
            num_keys: self.num_keys,
            data,
            _marker: std::marker::PhantomData,
        })
    }
}

macro_rules! remove_edge {
    ($xor_graph: ident, $e: ident, $side: ident, $edge: ident, $stack: ident, $push:ident, $conv: expr) => {
        match $side {
            0 => {
                if $xor_graph.degree($e[1]) == 2 {
                    $stack.$push($conv($e[1]));
                }
                $xor_graph.remove($e[1], $edge, 1);
                if $xor_graph.degree($e[2]) == 2 {
                    $stack.$push($conv($e[2]));
                }
                $xor_graph.remove($e[2], $edge, 2);
            }
            1 => {
                if $xor_graph.degree($e[0]) == 2 {
                    $stack.$push($conv($e[0]));
                }
                $xor_graph.remove($e[0], $edge, 0);
                if $xor_graph.degree($e[2]) == 2 {
                    $stack.$push($conv($e[2]));
                }
                $xor_graph.remove($e[2], $edge, 2);
            }
            2 => {
                if $xor_graph.degree($e[0]) == 2 {
                    $stack.$push($conv($e[0]));
                }
                $xor_graph.remove($e[0], $edge, 0);
                if $xor_graph.degree($e[1]) == 2 {
                    $stack.$push($conv($e[1]));
                }
                $xor_graph.remove($e[1], $edge, 1);
            }
            // SAFETY: side is always 0, 1, or 2 (encoded as a 2-bit
            // field in the degrees_sides array).
            _ => unsafe { unreachable_unchecked() },
        }
    };
}

impl<
    W: Word + BinSafe + Send + Sync,
    D: BitFieldSlice<Value = W>
        + for<'a> BitFieldSliceMut<Value = W, ChunksMut<'a>: Iterator<Item: BitFieldSliceMut>>
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
    fn par_solve<
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
            .expected_updates(Some(self.shard_edge.num_shards()))
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
                for val in shard_iter
                    .into_iter()
                    .zip(data.try_chunks_mut(self.shard_edge.num_vertices()).unwrap())
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

                                main_pl.info(format_args!(
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

                                main_pl.info(format_args!(
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

                                main_pl.info(format_args!(
                                    "Completed shard {}/{}",
                                    shard_index + 1,
                                    num_shards
                                ));
                                main_pl.update_and_display();
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
    /// This peeler uses more memory than
    /// [`peel_by_sig_vals_low_mem`](VBuilder::peel_by_sig_vals_low_mem) but
    /// less memory than
    /// [`peel_by_sig_vals_high_mem`](VBuilder::peel_by_sig_vals_high_mem). It
    /// is fairly slow as it has to go through a cache-unfriendly memory
    /// indirection every time it has to retrieve a [`SigVal`] from the shard,
    /// but it is the peeler of choice when [lazy Gaussian
    /// elimination](https://doi.org/10.1016/j.ic.2020.104517) is required, as
    /// after a failed peel-by-sig-vals it is not possible to retrieve
    /// information about the signature/value pairs in the shard.
    ///
    /// In theory one could avoid the stack of sides by putting vertices,
    /// instead of edge indices, on the upper stack, and retrieving edge indices
    /// and sides from the [`XorGraph`], as
    /// [`peel_by_sig_vals_low_mem`](VBuilder::peel_by_sig_vals_low_mem) does,
    ///  but this would be less cache friendly. This peeler is only used for
    /// very small instances, and since we are going to pass through lazy
    /// Gaussian elimination some additional speed is a good idea.
    fn peel_by_index<
        'a,
        V: BinSafe,
        G: Fn(&E, SigVal<E::LocalSig, V>) -> W + Send + Sync,
        H: Fn(&SigVal<S, V>) + Send + Sync,
    >(
        &self,
        shard_index: usize,
        shard: Arc<Vec<SigVal<S, V>>>,
        data: ShardData<'a, D>,
        get_val: &G,
        inspect: &H,
        pl: &mut impl ProgressLog,
    ) -> Result<PeelResult<'a, W, D, S, E, V>, ()> {
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
            pl.info(format_args!(
                "Peeling failed for shard {}/{} (peeled {} out of {} edges)",
                shard_index + 1,
                num_shards,
                double_stack.upper_len(),
                shard.len(),
            ));
            return Ok(PeelResult::Partial {
                shard_index,
                shard,
                data,
                double_stack,
                sides_stack,
                _marker: PhantomData,
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
    /// [`peel_by_sig_vals_low_mem`](VBuilder::peel_by_sig_vals_low_mem), which
    /// uses almost half the memory. It is the peeler of choice for low levels
    /// of parallelism.
    ///
    /// This peeler cannot be used in conjunction with [lazy Gaussian
    /// elimination](https://doi.org/10.1016/j.ic.2020.104517) as after a failed
    /// peeling it is not possible to retrieve information about the
    /// signature/value pairs in the shard.
    fn peel_by_sig_vals_high_mem<
        V: BinSafe,
        G: Fn(&E, SigVal<E::LocalSig, V>) -> W + Send + Sync,
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

        if shard_len != sig_vals_stack.len() {
            pl.info(format_args!(
                "Peeling failed for shard {}/{} (peeled {} out of {} edges)",
                shard_index + 1,
                num_shards,
                sig_vals_stack.len(),
                shard_len
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
    /// slower than
    /// [`peel_by_sig_vals_high_mem`](VBuilder::peel_by_sig_vals_high_mem),
    /// which uses almost twice the memory. It is the peeler of choice for
    /// significant levels of parallelism.
    ///
    /// This peeler cannot be used in conjunction with [lazy Gaussian
    /// elimination](https://doi.org/10.1016/j.ic.2020.104517) as after a
    /// failed peeling it is not possible to retrieve information about the
    /// signature/value pairs in the shard.
    fn peel_by_sig_vals_low_mem<
        V: BinSafe,
        G: Fn(&E, SigVal<E::LocalSig, V>) -> W + Send + Sync,
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
            pl.info(format_args!(
                "Peeling failed for shard {}/{} (peeled {} out of {} edges)",
                shard_index + 1,
                num_shards,
                visit_stack.upper_len(),
                shard_len
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
    /// elimination](https://doi.org/10.1016/j.ic.2020.104517), and stores the
    /// solution in the given data.
    ///
    /// As a first try, the shard is [peeled by index](VBuilder::peel_by_index).
    /// If the peeling is [partial](PeelResult::Partial), lazy Gaussian
    /// elimination is used to solve the remaining edges.
    ///
    /// This method will scan the double stack, without emptying it, to check
    /// which edges have been peeled. The information will be then passed to
    /// [`VBuilder::assign`] to complete the assignment of values.
    fn lge_shard<
        V: BinSafe,
        G: Fn(&E, SigVal<E::LocalSig, V>) -> W + Send + Sync,
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
                _marker: PhantomData,
            }) => {
                pl.info(format_args!("Switching to lazy Gaussian elimination..."));
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
        sigs_vals_sides: impl Iterator<Item = ((E::LocalSig, W), u8)>,
        pl: &mut impl ProgressLog,
    ) where
        for<'a> ShardData<'a, D>: SliceByValueMut<Value = W>,
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
