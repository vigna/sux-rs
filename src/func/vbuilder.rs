/*
 * SPDX-FileCopyrightText: 2025 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]

use crate::bits::*;
use crate::func::{shard_edge::ShardEdge, *};
use crate::traits::bit_field_slice::{BitFieldSlice, BitFieldSliceMut};
use crate::traits::{BitVecOpsMut, Word};
use crate::utils::*;
use core::error::Error;
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
use std::ops::{BitXor, BitXorAssign};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;
use sync_cell_slice::SyncSlice;
use thread_priority::ThreadPriority;
use value_traits::slices::{SliceByValue, SliceByValueMut};

use super::shard_edge::{Fuse3Shards, FuseAug3Shards, LOG2_MAX_SHARDS};

/// Returns the default maximum number of threads: `min(16, available_parallelism)`.
fn default_max_num_threads() -> usize {
    std::thread::available_parallelism()
        .map(|p| p.get().min(16))
        .unwrap_or(1)
}

/// A builder for static functions, static filters, and the structures based
/// on them.
///
/// A [`VBuilder`] holds the construction parameters of [`VFunc`], [`VFunc2`],
/// [`CompVFunc`], [`VFilter`], [`LcpMmphfInt`], [`LcpMmphf`],
/// [`Lcp2MmphfInt`], [`Lcp2Mmphf`], and of the [`SignedFunc`] wrappers around
/// them. You pass it to the `*_with_builder` constructors of these structures
/// (e.g., [`VFunc::try_new_with_builder`] or
/// [`VFunc::try_par_new_with_builder`]); the other constructors use
/// [`VBuilder::default()`].
///
/// The setters are:
///
/// - [`offline`]: stores signatures and values in on-disk buckets instead of
///   core memory; only sequential constructors support this option;
/// - [`temp_dir`]: the directory of the on-disk buckets of offline mode;
/// - [`eps`]: the target relative space loss due to ε-cost sharding;
/// - [`retry_prob`]: the maximum probability of a retry caused by duplicate
///   edges;
/// - [`low_mem`]: forces the low-memory (`true`) or the high-memory (`false`)
///   peeler; by default, the builder chooses depending on the number of
///   threads and shards;
/// - [`check_dups`]: checks explicitly for duplicate keys;
/// - [`seed`]: the seed of the generator of the hashing seeds;
/// - [`max_num_threads`]: the maximum number of threads;
/// - [`expected_num_keys`]: an estimate of the number of keys, which speeds up
///   the construction;
/// - [`log2_buckets`]: the base-2 logarithm of the number of buckets of the
///   [`SigStore`].
///
/// # Examples
///
/// ```rust
/// # #[cfg(feature = "rayon")]
/// # fn main() -> anyhow::Result<()> {
/// # use sux::func::{VBuilder, VFunc};
/// # use sux::bits::BitFieldVec;
/// # use dsi_progress_logger::no_logging;
/// # use sux::utils::FromCloneableIntoIterator;
/// let n = 1000;
/// let func = <VFunc<usize, BitFieldVec<Box<[usize]>>>>::try_new_with_builder(
///     FromCloneableIntoIterator::new(0..n),
///     FromCloneableIntoIterator::new(0..n),
///     VBuilder::default()
///         .offline(true)
///         .expected_num_keys(n)
///         .max_num_threads(4),
///     no_logging![],
/// )?;
///
/// for i in 0..n {
///     assert_eq!(func.get(i), i);
/// }
/// # Ok(())
/// # }
/// # #[cfg(not(feature = "rayon"))]
/// # fn main() {}
/// ```
///
/// # Implementation details
///
/// Initially, keys are scanned, turned into signatures, and stored in a
/// [`SigStore`] (in core memory or, in [offline] mode, in on-disk buckets),
/// possibly together with associated values. Sequential constructors take
/// [`FallibleRewindableLender`]s, as the construction might fail and keys
/// might be scanned again; parallel constructors take slices and compute the
/// signatures in parallel.
///
/// Once signatures have been computed, each parallel thread will process a
/// shard of the signature/value pairs. For very large key sets shards will be
/// significantly smaller than the number of keys, so the memory usage, in
/// particular in offline mode, can be significantly reduced. Note that using
/// too many threads might actually be harmful due to memory contention.
///
/// The generic parameters are explained in the [`VFunc`] documentation.
///
/// [`offline`]: VBuilder::offline
/// [offline]: VBuilder::offline
/// [`temp_dir`]: VBuilder::temp_dir
/// [`eps`]: VBuilder::eps
/// [`retry_prob`]: VBuilder::retry_prob
/// [`low_mem`]: VBuilder::low_mem
/// [`check_dups`]: VBuilder::check_dups
/// [`seed`]: VBuilder::seed
/// [`max_num_threads`]: VBuilder::max_num_threads
/// [`expected_num_keys`]: VBuilder::expected_num_keys
/// [`log2_buckets`]: VBuilder::log2_buckets
/// [`VFilter`]: crate::dict::VFilter
#[derive(Setters, Debug)]
#[setters(generate = false)]
pub struct VBuilder<D, S = [u64; 2], E = FuseAug3Shards> {
    /// The expected number of keys.
    ///
    /// While this setter is optional, setting this value to a reasonable bound
    /// on the actual number of keys will significantly speed up the
    /// construction.
    #[setters(generate = true, strip_option)]
    expected_num_keys: Option<usize>,

    /// Overrides the value used to size the sharding step.
    ///
    /// Normally [`set_up_shards`](ShardEdge::set_up_shards) is called
    /// with the number of keys, which is correct for a construction
    /// where each key produces exactly one hyperedge (VFunc). When
    /// each key produces *multiple* hyperedges (as in
    /// [`CompVFunc`], where each key's
    /// Huffman-encoded value contributes one edge per codeword bit),
    /// the graph actually has ≈ entropy × num_keys equations.
    /// Sizing the shards for the smaller key count leaves every
    /// shard oversized relative to the sharding strategy's intent.
    ///
    /// If this field is `Some(n)`, `n` is passed to
    /// [`set_up_shards`](ShardEdge::set_up_shards) instead of the
    /// runtime key count. CompVFunc sets this to the total edge
    /// count after building the Huffman coder.
    pub(crate) shard_size_hint: Option<usize>,

    /// The maximum number of parallel threads to use for both the population
    /// and solve phases. The default is `min(16, available_parallelism)`.
    #[setters(generate = true)]
    pub(crate) max_num_threads: usize,

    /// Uses disk-based buckets to reduce core memory usage at construction
    /// time. Only the sequential constructors honor this; the parallel
    /// constructors return an error if it is set.
    #[setters(generate = true)]
    offline: bool,

    /// The directory in which [offline] mode creates the directory of its
    /// buckets.
    ///
    /// By default, the default temporary directory (see
    /// [`tempfile::env::temp_dir`]; on Unix, the directory specified by the
    /// environment variable `TMPDIR`, if set). Libraries should use this
    /// setting to let their users choose where temporary files are written,
    /// as the default temporary directory can be changed only process-wide.
    ///
    /// [offline]: Self::offline
    #[setters(generate = true, strip_option, into)]
    temp_dir: Option<PathBuf>,

    /// Checks for duplicate signatures.
    ///
    /// This is not necessary in general, but if you suspect you might be
    /// feeding duplicate keys, you can enable this check: construction will
    /// then fail with [`BuildError::DuplicateKey`].
    #[setters(generate = true)]
    check_dups: bool,

    /// Always uses the low-memory peel-by-signature algorithm (true) or the
    /// high-memory peel-by-index algorithm (false). The former is slightly
    /// slower, but it uses much less memory. Normally [`VBuilder`] uses
    /// high-mem and switches to low-mem if there are more
    /// than three threads and more than two shards.
    #[setters(generate = true, strip_option)]
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
    log2_buckets: u32,

    /// The target relative space loss due to [ε-cost sharding].
    ///
    /// The value must lie in the open interval (0 . . 1); constructors return an
    /// error otherwise.
    ///
    /// The default is 0.001. Setting a larger target, for example, 0.01, will
    /// increase the space overhead due to sharding, but will provide in general
    /// finer shards. This might not always happen, however, because the ε-cost
    /// bound is just one of the bounds concurring in limiting the amount of
    /// sharding for a specific [`ShardEdge`]. For example, increasing the
    /// target to 0.01 will provide very fine sharding using `Mwhc3Shards`
    /// shard/edge logic, activated by the `mwhc` feature, but will affect only
    /// slightly [`FuseAug3Shards`] or [`FuseAug3FullSigs`].
    ///
    /// [`FuseAug3FullSigs`]: crate::func::shard_edge::FuseAug3FullSigs
    ///
    /// [ε-cost sharding]: https://arxiv.org/abs/2503.18397
    #[setters(generate = true, strip_option)]
    pub(crate) eps: f64,

    /// The maximum probability of a construction retry caused by duplicate
    /// hyperedges.
    ///
    /// Sharding is limited so that the probability that a shard contains two
    /// identical hyperedges, which forces a new construction attempt with a
    /// different seed, is at most this value. The bound does not cover
    /// duplicate local signatures, which can occur only with huge key sets
    /// and lead to [`BuildError::DuplicateLocalSignatures`] (see
    /// [`FuseAug3FullSigs`](crate::func::shard_edge::FuseAug3FullSigs)). The
    /// value must lie in the half-open interval (0 . . 1];
    /// constructors return an error otherwise.
    ///
    /// The default is 0.05. Setting a larger bound will in general provide
    /// finer shards at the price of a higher probability of retrying the
    /// construction with a new seed: with values close to one, construction
    /// might need many attempts, and fail if it exhausts them.
    #[setters(generate = true)]
    pub(crate) retry_prob: f64,

    /// The edge generator.
    pub(crate) shard_edge: E,
    /// The number of keys.
    pub(crate) num_keys: usize,
    /// The ratio between the number of vertices and the number of edges
    /// (i.e., between the number of variables and the number of equations).
    pub(crate) c: f64,
    /// Whether we should solve shards by augmented peeling (see
    /// [`ShardEdge::set_up_graphs`]).
    pub(crate) aug: bool,
    /// The number of threads to use.
    pub(crate) num_threads: usize,
    /// Fast-stop for failed attempts.
    pub(crate) failed: AtomicBool,
    #[doc(hidden)]
    _marker: PhantomData<(D, S)>,
}

impl<D, S, E: Default> Default for VBuilder<D, S, E> {
    fn default() -> Self {
        Self {
            expected_num_keys: None,
            shard_size_hint: None,
            max_num_threads: default_max_num_threads(),
            offline: false,
            temp_dir: None,
            check_dups: false,
            low_mem: None,
            seed: 0,
            log2_buckets: 8,
            eps: 0.001,
            retry_prob: Fuse3Shards::DUP_EDGE_RETRY_PROB,
            shard_edge: E::default(),
            num_keys: 0,
            c: 0.0,
            aug: false,
            num_threads: 0,
            failed: AtomicBool::new(false),
            _marker: PhantomData,
        }
    }
}

impl<D: BitFieldSlice<Value: Word + BinSafe> + Send + Sync, S: Sig, E: ShardEdge<S, 3>>
    VBuilder<D, S, E>
{
    /// Copies behavioral configuration from another builder into `self`,
    /// regardless of the other builder's type parameters.
    ///
    /// The copied fields are: [`max_num_threads`], [`offline`],
    /// [`temp_dir`], [`check_dups`], [`low_mem`], [`eps`], and
    /// [`retry_prob`].
    /// Data-dependent fields
    /// ([`expected_num_keys`], [`seed`], [`log2_buckets`]) and internal
    /// construction state are left at their defaults.
    ///
    /// [`max_num_threads`]: Self::max_num_threads
    /// [`offline`]: Self::offline
    /// [`temp_dir`]: Self::temp_dir
    /// [`check_dups`]: Self::check_dups
    /// [`low_mem`]: Self::low_mem
    /// [`eps`]: Self::eps
    /// [`retry_prob`]: Self::retry_prob
    /// [`expected_num_keys`]: Self::expected_num_keys
    /// [`seed`]: Self::seed
    /// [`log2_buckets`]: Self::log2_buckets
    pub fn set_from<D2, S2, E2>(mut self, other: &VBuilder<D2, S2, E2>) -> Self {
        self.max_num_threads = other.max_num_threads;
        self.offline = other.offline;
        self.temp_dir = other.temp_dir.clone();
        self.check_dups = other.check_dups;
        self.low_mem = other.low_mem;
        self.eps = other.eps;
        self.retry_prob = other.retry_prob;
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
    /// The largest shard was too large with respect to the average shard
    /// size on 100 construction attempts; a larger [ε](VBuilder::eps) might
    /// help.
    #[error("Max shard too big after multiple attempts: try a larger eps")]
    MaxShardTooBig,
    /// A shard was unsolvable on 100 construction attempts (1000 when using
    /// augmented peeling).
    #[error("Unsolvable shard after multiple attempts")]
    UnsolvableShard,
    /// The number of values is different from the number of keys.
    ///
    /// Slice-based constructors compare the lengths exactly. Lender-based
    /// constructors detect the mismatch lazily, so the reported counts might
    /// be lower bounds; moreover, they detect only a values lender that is
    /// exhausted before the keys: by contract, a values lender may return
    /// more values than keys (in particular, it may be infinite), and the
    /// extra values are ignored.
    #[error("Mismatch between number of keys ({num_keys}) and number of values ({num_values})")]
    MismatchedKeysAndValues {
        /// The number of keys.
        num_keys: usize,
        /// The number of values.
        num_values: usize,
    },
}

/// Transient error during the build, leading to trying with a different seed.
#[derive(thiserror::Error, Debug)]
pub(crate) enum SolveError {
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

/// Returns [`BuildError::MismatchedKeysAndValues`] if the number of keys
/// and the number of values of a slice-based constructor differ.
pub(crate) fn check_num_values(num_keys: usize, num_values: usize) -> anyhow::Result<()> {
    if num_keys != num_values {
        return Err(BuildError::MismatchedKeysAndValues {
            num_keys,
            num_values,
        }
        .into());
    }
    Ok(())
}

/// Returns `true` when the largest shard deviates pathologically from the
/// target ε-cost. Both the sequential and parallel build paths use this
/// predicate so their acceptance thresholds cannot drift apart.
///
/// The tolerance is the maximum of the ε-cost term `5·eps` times the
/// average shard size and the statistical term `2.5·√(avg·ln S)`, which
/// covers the natural fluctuation of the maximum of `S` multinomial
/// shard sizes (about `√(2·avg·ln S)` above the average).
fn max_shard_too_big(max_shard: usize, num_keys: usize, num_shards: usize, eps: f64) -> bool {
    // f64 comparison; the usize->f64 precision loss is irrelevant at
    // these magnitudes and matches the surrounding cost model.
    let avg = num_keys as f64 / num_shards as f64;
    let tolerance = (5.0 * eps * avg).max(2.5 * (avg * (num_shards as f64).ln().max(1.0)).sqrt());
    max_shard as f64 > avg + tolerance
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
    pub(crate) fn try_build_func_with_store<
        K: ?Sized + ToSig<S>,
        V: BinSafe + Default + Send + Sync,
    >(
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
    /// [`try_build_func_with_store`]: Self::try_build_func_with_store
    fn try_build_func_with_store_and_inspect<
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

        // The shard structure is fixed by the store (set at into_shard_store
        // time). We only reconfigure graph parameters (c, aug, segment size,
        // vertex count) for the actual key count and shard sizes.
        let max_shard = shard_store.shard_sizes().max().unwrap_or(0);
        (self.c, self.aug) = self.shard_edge.set_up_graphs(self.num_keys, max_shard);

        pl.info(format_args!("Max value: {max_value}"));

        self.try_build_from_shard_iter(
            seed,
            max_value.bit_len() as usize,
            BitFieldVec::new_padded,
            shard_store.iter(),
            get_val,
            inspect,
            pl,
        )
        .map_err(Into::into)
    }

    /// Builds the first function of a two-step construction (see
    /// [`VFunc2`]) from a populated [`ShardStore`].
    ///
    /// The function maps each key to `get_val` of its value, and is built
    /// as in [`try_build_func_with_store`], with the same preconditions.
    /// Besides the function, this method returns, for each shard of the
    /// store at its [maximum granularity], the number of keys whose value
    /// satisfies `escaped`; [`try_build_escaped_func`] uses the counts to
    /// build the second function.
    ///
    /// [`try_build_func_with_store`]: Self::try_build_func_with_store
    /// [`try_build_escaped_func`]: Self::try_build_escaped_func
    /// [maximum granularity]: ShardStore::max_shard_high_bits
    pub(crate) fn try_build_first_func<K: ?Sized + ToSig<S>, V: BinSafe + Default + Send + Sync>(
        &mut self,
        seed: u64,
        shard_edge: E,
        max_value: W,
        store: &mut (impl ShardStore<S, V> + ?Sized),
        get_val: &(impl Fn(V) -> W + Send + Sync),
        escaped: &(impl Fn(V) -> bool + Send + Sync),
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> anyhow::Result<(VFunc<K, BitFieldVec<Box<[W]>>, S, E>, Vec<usize>)>
    where
        SigVal<S, V>: RadixKey,
        SigVal<E::LocalSig, V>: BitXor + BitXorAssign,
    {
        // The counters are kept at maximum granularity, so they can be
        // re-aggregated to the sharding of the second function.
        let max_shard_high_bits = store.max_shard_high_bits();
        let max_shard_mask = (1u64 << max_shard_high_bits) - 1;
        let mut escaped_counts = vec![0usize; 1 << max_shard_high_bits];
        let sync_counts = escaped_counts.as_sync_slice();

        let func = self.try_build_func_with_store_and_inspect(
            seed,
            shard_edge,
            max_value,
            store,
            &|_, sig_val| get_val(sig_val.val),
            &|sig_val: &SigVal<S, V>| {
                if escaped(sig_val.val) {
                    let shard = sig_val.sig.high_bits(max_shard_high_bits, max_shard_mask) as usize;
                    // SAFETY: each shard is processed by exactly one
                    // thread, so no two threads access the same counter.
                    unsafe {
                        let count = sync_counts[shard].get();
                        sync_counts[shard].set(count + 1);
                    }
                }
            },
            pl,
        )?;

        Ok((func, escaped_counts))
    }

    /// Builds the second function of a two-step construction (see
    /// [`VFunc2`]) on the keys of a populated [`ShardStore`] whose value
    /// satisfies `escaped`.
    ///
    /// The function maps each such key to `get_val` of its value.
    /// `escaped_counts` must be the counts returned by
    /// [`try_build_first_func`] for the same store and predicate.
    ///
    /// The sharding of the function is set up for the number of escaped keys
    /// using the [ε] and the [retry probability] of this builder, and the
    /// function is built by a builder with the [same configuration].
    ///
    /// [`try_build_first_func`]: Self::try_build_first_func
    /// [ε]: Self::eps
    /// [retry probability]: Self::retry_prob
    /// [same configuration]: Self::set_from
    pub(crate) fn try_build_escaped_func<
        K: ?Sized + ToSig<S>,
        V: BinSafe + Default + Send + Sync,
        F: ShardEdge<S, 3> + MemSize + FlatType,
    >(
        &self,
        seed: u64,
        store: &mut (impl ShardStore<S, V> + ?Sized),
        escaped_counts: &[usize],
        escaped: &(impl Fn(V) -> bool + Send + Sync),
        max_value: W,
        get_val: &(impl Fn(V) -> W + Send + Sync),
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> anyhow::Result<VFunc<K, BitFieldVec<Box<[W]>>, S, F>>
    where
        SigVal<S, V>: RadixKey,
        SigVal<F::LocalSig, V>: BitXor + BitXorAssign,
        SigVal<F::LocalSig, W>: BitXor + BitXorAssign,
    {
        let mut shard_edge = F::default();
        shard_edge.set_up_shards(escaped_counts.iter().sum(), self.eps, self.retry_prob);
        let shard_high_bits = shard_edge.shard_high_bits();

        // Aggregate the counters to the sharding of the function
        let filtered_shard_sizes: Vec<usize> = escaped_counts
            .chunks(escaped_counts.len() >> shard_high_bits)
            .map(|chunk| chunk.iter().sum())
            .collect();

        let mut filtered_store = FilteredShardStore::new(
            store,
            shard_high_bits,
            |sig_val: &SigVal<S, V>| escaped(sig_val.val),
            filtered_shard_sizes,
        );

        VBuilder::<BitFieldVec<Box<[W]>>, S, F>::default()
            .set_from(self)
            .try_build_func_with_store(
                seed,
                shard_edge,
                max_value,
                &mut filtered_store,
                &|_, sig_val| get_val(sig_val.val),
                pl,
            )
    }
}

/// State for the retry loop of [`VBuilder::try_with_retries`].
///
/// Create with [`VBuilder::retry_state`], then call
/// [`handle_solve_result`] after each attempt.
///
/// [`handle_solve_result`]: RetryState::handle_solve_result
struct RetryState {
    prng: SmallRng,
    dup_count: u32,
    local_dup_count: u32,
    max_shard_too_big: u32,
    unsolvable_count: u32,
}

impl RetryState {
    /// The number of attempts after which a [`SolveError::MaxShardTooBig`]
    /// or a [`SolveError::UnsolvableShard`] becomes fatal.
    const MAX_ATTEMPTS: u32 = 100;

    /// The number of attempts after which a [`SolveError::UnsolvableShard`]
    /// becomes fatal when using augmented peeling.
    ///
    /// Augmented peeling is used on small shards, whose graphs are sized so
    /// that a construction attempt succeeds with probability about 70%; the
    /// larger bound is a safety margin for tiny key sets, on which the
    /// estimate of the success probability is less accurate.
    const MAX_AUG_ATTEMPTS: u32 = 1000;

    /// Returns the next seed to try.
    fn next_seed(&mut self) -> u64 {
        self.prng.random()
    }

    /// Handles the result of a construction attempt; `aug` tells whether the
    /// attempt used augmented peeling.
    ///
    /// On success, returns `Ok(Some(r))`. On retryable error, logs a
    /// warning and returns `Ok(None)` (the caller should rewind its
    /// lenders and retry). On fatal error, returns `Err(e)`.
    fn handle_solve_result<R>(
        &mut self,
        result: anyhow::Result<R>,
        aug: bool,
        pl: &mut impl ProgressLog,
    ) -> anyhow::Result<Option<R>> {
        match result {
            Ok(r) => Ok(Some(r)),
            Err(error) => match error.downcast::<SolveError>() {
                Ok(vfunc_error) => match vfunc_error {
                    SolveError::DuplicateSignature => {
                        if self.dup_count >= 2 {
                            pl.error(format_args!("Duplicate keys (duplicate 128-bit signatures with three different seeds)"));
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
                        self.max_shard_too_big += 1;
                        if self.max_shard_too_big >= Self::MAX_ATTEMPTS {
                            pl.error(format_args!(
                                "The maximum shard was too big on {} attempts",
                                Self::MAX_ATTEMPTS
                            ));
                            return Err(BuildError::MaxShardTooBig.into());
                        }
                        pl.warn(format_args!(
                            "The maximum shard is too big, trying again with a different seed..."
                        ));
                        Ok(None)
                    }
                    SolveError::UnsolvableShard => {
                        self.unsolvable_count += 1;
                        let max_attempts = if aug {
                            Self::MAX_AUG_ATTEMPTS
                        } else {
                            Self::MAX_ATTEMPTS
                        };
                        if self.unsolvable_count >= max_attempts {
                            pl.error(format_args!(
                                "A shard was unsolvable on {max_attempts} attempts"
                            ));
                            return Err(BuildError::UnsolvableShard.into());
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
        self.try_populate_and_build(
            keys,
            values,
            &mut |builder, seed, mut store, max_value, _num_keys, pl: &mut P, _state: &mut ()| {
                pl.info(format_args!("Max value: {max_value}"));
                builder
                    .try_build_from_shard_iter(
                        seed,
                        max_value.bit_len() as usize,
                        new_data,
                        store.drain(),
                        &|_, sig_val| sig_val.val,
                        &|_| {},
                        pl,
                    )
                    .map_err(Into::into)
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
        self.try_populate_and_build(
            keys,
            values,
            &mut |builder, seed, mut store, max_value, _num_keys, pl: &mut P, _state: &mut ()| {
                pl.info(format_args!("Max value: {max_value}"));
                let func = builder.try_build_from_shard_iter(
                    seed,
                    max_value.bit_len() as usize,
                    new_data,
                    store.iter(),
                    &|_, sig_val| sig_val.val,
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

    /// Builds a new [`VFunc`] from in-memory keys, computing signatures in
    /// parallel.
    ///
    /// This is the parallel counterpart of [`try_build_func`]: the value
    /// associated with `keys[i]` is `val_fn(i)`.
    ///
    /// [`try_build_func`]: Self::try_build_func
    pub(crate) fn try_par_build_func<
        K: ?Sized + ToSig<S> + std::fmt::Debug + Sync,
        B: Borrow<K> + Sync,
        P: ProgressLog + Clone + Send + Sync,
    >(
        mut self,
        keys: &[B],
        val_fn: &(impl Fn(usize) -> D::Value + Send + Sync),
        new_data: fn(usize, usize) -> D,
        pl: &mut P,
    ) -> anyhow::Result<VFunc<K, D, S, E>>
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
        self.expected_num_keys = Some(keys.len());
        self.try_par_populate_and_build(
            keys,
            val_fn,
            &mut |builder, seed, mut store, max_value, _num_keys, pl: &mut P, _state: &mut ()| {
                pl.info(format_args!("Max value: {max_value}"));
                builder
                    .try_build_from_shard_iter(
                        seed,
                        max_value.bit_len() as usize,
                        new_data,
                        store.drain(),
                        &|_, sig_val| sig_val.val,
                        &|_| {},
                        pl,
                    )
                    .map_err(Into::into)
            },
            pl,
            (),
        )
    }

    /// Builds a [`VFunc`] suitable for use as a filter backend.
    ///
    /// Unlike [`try_build_func`], this method takes only keys (no values),
    /// uses a caller-specified `bit_width` instead of deriving it from the
    /// maximum value, and derives stored values from signatures via
    /// `get_val`.
    ///
    /// `new_data(bit_width, len)` allocates the backend storage.
    ///
    /// [`try_build_func`]: Self::try_build_func
    pub(crate) fn try_build_filter<
        K: ?Sized + ToSig<S> + std::fmt::Debug,
        B: ?Sized + Borrow<K>,
        P: ProgressLog + Clone + Send + Sync,
        G: Fn(&E, SigVal<E::LocalSig, EmptyVal>) -> D::Value + Send + Sync,
    >(
        mut self,
        keys: impl FallibleRewindableLender<
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
        self.try_populate_and_build(
            keys,
            FromCloneableIntoIterator::new(std::iter::repeat(EmptyVal::default())),
            &mut |builder, seed, mut store, _max_value, _num_keys, pl: &mut P, _state: &mut ()| {
                builder
                    .try_build_from_shard_iter(
                        seed,
                        bit_width,
                        new_data,
                        store.drain(),
                        get_val,
                        &|_| {},
                        pl,
                    )
                    .map_err(Into::into)
            },
            pl,
            (),
        )
        .map(|(func, _keys)| func)
    }

    /// Builds a [`VFunc`] suitable for use as a filter backend from
    /// in-memory keys, computing signatures in parallel.
    ///
    /// This is the parallel counterpart of [`try_build_filter`].
    ///
    /// [`try_build_filter`]: Self::try_build_filter
    pub(crate) fn try_par_build_filter<
        K: ?Sized + ToSig<S> + std::fmt::Debug + Sync,
        B: Borrow<K> + Sync,
        P: ProgressLog + Clone + Send + Sync,
        G: Fn(&E, SigVal<E::LocalSig, EmptyVal>) -> D::Value + Send + Sync,
    >(
        mut self,
        keys: &[B],
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
        self.expected_num_keys = Some(keys.len());
        self.try_par_populate_and_build(
            keys,
            &|_| EmptyVal::default(),
            &mut |builder, seed, mut store, _max_value, _num_keys, pl: &mut P, _state: &mut ()| {
                builder
                    .try_build_from_shard_iter(
                        seed,
                        bit_width,
                        new_data,
                        store.drain(),
                        get_val,
                        &|_| {},
                        pl,
                    )
                    .map_err(Into::into)
            },
            pl,
            (),
        )
    }

    /// Validates the configuration, sets up shards from the expected number
    /// of keys, and returns a [`RetryState`] for driving the retry loop.
    fn retry_state(&mut self, pl: &mut impl ProgressLog) -> anyhow::Result<RetryState> {
        // The ε-cost formulas divide by ln(1 - eps): eps == 0 divides by
        // zero and eps >= 1 yields NaN, so only the open interval is
        // meaningful. The range check also rejects NaN and ±∞.
        anyhow::ensure!(
            self.eps > 0.0 && self.eps < 1.0,
            "eps must be in the open interval (0..1), got {}",
            self.eps
        );
        // A retry probability of one is meaningful (accept any amount of
        // duplicate-edge risk), but zero would forbid sharding entirely and
        // the comparisons below would reject NaN silently.
        anyhow::ensure!(
            self.retry_prob > 0.0 && self.retry_prob <= 1.0,
            "retry_prob must be in the half-open interval (0..1], got {}",
            self.retry_prob
        );
        if let Some(expected_num_keys) = self.expected_num_keys {
            self.shard_edge
                .set_up_shards(expected_num_keys, self.eps, self.retry_prob);
            self.log2_buckets = self.shard_edge.shard_high_bits();
        }
        pl.info(format_args!(
            "Using a store with 2^{} buckets",
            self.log2_buckets
        ));
        Ok(RetryState {
            prng: SmallRng::seed_from_u64(self.seed),
            dup_count: 0,
            local_dup_count: 0,
            max_shard_too_big: 0,
            unsolvable_count: 0,
        })
    }

    /// Runs construction attempts with different seeds until one succeeds.
    ///
    /// This is a low-level method that requires a thorough understanding
    /// of the builder's internal state.
    ///
    /// After validating the configuration and setting up shards from the
    /// expected number of keys, this method calls `attempt(self, seed,
    /// &mut state, pl)` with seeds drawn from a generator initialized with
    /// the [builder seed]. Retries continue for [`SolveError::UnsolvableShard`]
    /// and [`SolveError::MaxShardTooBig`] (up to 100 attempts, or 1000
    /// attempts for [`SolveError::UnsolvableShard`] when using augmented
    /// peeling, after which [`BuildError::UnsolvableShard`] or
    /// [`BuildError::MaxShardTooBig`], respectively, is returned); if the
    /// third attempt still finds a
    /// duplicate signature, [`BuildError::DuplicateKey`] is returned, and if
    /// it still finds a duplicate local signature,
    /// [`BuildError::DuplicateLocalSignatures`] is returned. Any other error
    /// is propagated immediately.
    ///
    /// Before each retry, `state` is replaced with `rewind(state)`;
    /// usually, `state` contains the lenders to rewind.
    ///
    /// Returns the result of the successful attempt and the state.
    ///
    /// [builder seed]: Self::seed
    pub(crate) fn try_with_retries<T, R, P: ProgressLog>(
        &mut self,
        mut state: T,
        mut attempt: impl FnMut(&mut Self, u64, &mut T, &mut P) -> anyhow::Result<R>,
        mut rewind: impl FnMut(T) -> anyhow::Result<T>,
        pl: &mut P,
    ) -> anyhow::Result<(R, T)> {
        let mut rs = self.retry_state(pl)?;
        let start = Instant::now();

        loop {
            let seed = rs.next_seed();
            let result = attempt(self, seed, &mut state, pl);

            // self.aug has been set by the attempt
            if let Some(r) = rs.handle_solve_result(result, self.aug, pl)? {
                pl.info(format_args!(
                    "Construction completed in {:.3} seconds ({} keys, {:.3} ns/key)",
                    start.elapsed().as_secs_f64(),
                    self.num_keys,
                    start.elapsed().as_nanos() as f64 / self.num_keys as f64
                ));
                return Ok((r, state));
            }

            state = rewind(state)?;
        }
    }

    /// Populates a shard store from keys and values, then calls a build
    /// closure. If the closure (or the store population) fails with a
    /// [`SolveError`], the process is retried from scratch with a new
    /// seed, as described in [`try_with_retries`].
    ///
    /// This is a low-level method that requires a thorough understanding
    /// of the builder's internal state.
    ///
    /// On each retry the lenders are rewound.
    ///
    /// `build_fn` is called with `(&mut self, seed, store, max_value, num_keys,
    /// pl)`. The builder's `shard_edge`, `c`, and `aug` fields are already set
    /// up when `build_fn` is invoked, so it can call
    /// `try_build_from_shard_iter` directly.
    ///
    /// Returns whatever `build_fn` returns on success.
    ///
    /// [`try_with_retries`]: Self::try_with_retries
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
        keys: L,
        values: impl FallibleRewindableLender<
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
        self.try_with_retries(
            (keys, values),
            |builder, seed, (keys, values), pl| {
                let mut populate = |seed: u64,
                                    push: &mut dyn FnMut(SigVal<S, V>) -> anyhow::Result<()>,
                                    pl: &mut P,
                                    _state: &mut C| {
                    let mut maybe_max_value = V::default();
                    let mut num_keys = 0;
                    while let Some(key) = keys.next()? {
                        pl.light_update();
                        num_keys += 1;
                        let &maybe_val =
                            values.next()?.ok_or(BuildError::MismatchedKeysAndValues {
                                num_keys,
                                num_values: num_keys - 1,
                            })?;
                        maybe_max_value = Ord::max(maybe_max_value, maybe_val);
                        push(SigVal {
                            sig: K::to_sig(key.borrow(), seed),
                            val: maybe_val,
                        })?;
                    }
                    Ok(maybe_max_value)
                };

                builder.try_solve_once(seed, &mut populate, build_fn, pl, &mut state)
            },
            |(keys, values)| Ok((keys.rewind()?, values.rewind()?)),
            pl,
        )
        .map(|(r, (keys, _values))| (r, keys))
    }

    /// Like [`try_populate_and_build`], but takes key and value slices and
    /// parallelizes the hash computation and store population.
    ///
    /// This method calls [`SigStoreImpl::par_populate`] to compute signatures and
    /// populate the store in parallel on rayon worker threads.
    ///
    /// [`try_populate_and_build`]: Self::try_populate_and_build
    /// [`SigStoreImpl::par_populate`]: crate::utils::sig_store::SigStoreImpl::par_populate
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
        anyhow::ensure!(
            !self.offline,
            "offline mode is not supported by the parallel builder; use the sequential constructor (try_new_with_builder)"
        );
        let n = keys.len();

        self.try_with_retries(
            (),
            |builder, seed, _, pl| {
                let mut sig_store = sig_store::new_online::<S, V>(
                    builder.log2_buckets,
                    LOG2_MAX_SHARDS,
                    builder.expected_num_keys,
                )?;

                pl.expected_updates(n);
                pl.item_name("key");
                pl.start(format!(
                    "Computing and storing {}-bit signatures in parallel ({} threads) in RAM using seed 0x{seed:016x}...",
                    std::mem::size_of::<S>() * 8,
                    builder.max_num_threads,
                ));

                let start = Instant::now();

                let max_value =
                    sig_store.par_populate(n, builder.max_num_threads, |i| SigVal {
                        sig: K::to_sig(keys[i].borrow(), seed),
                        val: val_fn(i),
                    })?;

                pl.done();

                builder.try_build_from_sig_store(
                    seed, sig_store, max_value, start, build_fn, pl, &mut state,
                )
            },
            |()| Ok(()),
            pl,
        )
        .map(|(r, ())| r)
    }

    /// Performs a single population-and-build attempt for the given seed.
    ///
    /// Creates a [`SigStore`] (offline or online depending on
    /// `self.offline`), populates it via the `populate` closure, and
    /// completes the attempt as described in [`try_build_from_sig_store`].
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
    ///
    /// [`try_build_from_sig_store`]: Self::try_build_from_sig_store
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
                match &self.temp_dir {
                    Some(dir) => sig_store::new_offline_in::<S, V>(
                        dir,
                        self.log2_buckets,
                        LOG2_MAX_SHARDS,
                        self.expected_num_keys,
                    )?,
                    None => sig_store::new_offline::<S, V>(
                        self.log2_buckets,
                        LOG2_MAX_SHARDS,
                        self.expected_num_keys,
                    )?,
                },
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
    /// This is generic over `SS` so that the `populate` closure can push
    /// to the concrete store type without dynamic dispatch. The `state`
    /// parameter is forwarded to both `populate` and `build_fn`.
    ///
    /// [`try_solve_once`]: Self::try_solve_once
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

        let max_value = populate(
            seed,
            &mut |sig_val| sig_store.try_push(sig_val).map_err(Into::into),
            pl,
            state,
        )?;

        pl.done();

        self.try_build_from_sig_store(seed, sig_store, max_value, start, build_fn, pl, state)
    }

    /// Completes a construction attempt whose signature/value pairs have
    /// been stored in `sig_store`, starting at time `start`.
    ///
    /// Converts the store into a [`ShardStore`] sharded by `self.shard_edge`
    /// (sized for [`shard_size_hint`], if set, rather than for the number of
    /// keys), rejects the attempt with [`SolveError::MaxShardTooBig`] if the
    /// largest shard deviates pathologically from the average, sets up
    /// `self.shard_edge`, `self.c`, `self.aug`, and `self.num_keys`, boxes
    /// the store into a `dyn` [`ShardStore`], and calls `build_fn`.
    ///
    /// [`shard_size_hint`]: Self::shard_size_hint
    fn try_build_from_sig_store<
        V: BinSafe + Default + Send + Sync + Ord,
        R,
        P: ProgressLog + Clone + Send + Sync,
        SS: SigStore<S, V, ShardStore: 'static>,
        C,
    >(
        &mut self,
        seed: u64,
        sig_store: SS,
        max_value: V,
        start: Instant,
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
        let num_keys = sig_store.len();

        pl.info(format_args!(
            "Computation of signatures from inputs completed in {:.3} seconds ({} keys, {:.3} ns/key)",
            start.elapsed().as_secs_f64(),
            num_keys,
            start.elapsed().as_nanos() as f64 / num_keys as f64
        ));

        // When shard_size_hint is set (e.g. by CompVFunc with the total
        // number of edges after building the Huffman coder), the sharding
        // strategy is sized for the hinted workload rather than the raw key
        // count.
        let shard_n = self.shard_size_hint.unwrap_or(num_keys);
        let shard_edge = &mut self.shard_edge;
        shard_edge.set_up_shards(shard_n, self.eps, self.retry_prob);

        let shard_store = sig_store.into_shard_store(shard_edge.shard_high_bits())?;
        let max_shard = shard_store.shard_sizes().max().unwrap_or(0);

        if shard_edge.shard_high_bits() != 0 {
            pl.info(format_args!(
                "Max shard / average shard: {:.3}%",
                (100.0 * max_shard as f64) / (num_keys as f64 / shard_edge.num_shards() as f64)
            ));
        }

        // Reject pathological deviation from the target ε-cost. When
        // shard_size_hint is set the sharding is sized for the hinted
        // workload, so the per-key average is meaningless and the check is
        // skipped (CompVFunc performs its own check in edge units).
        if self.shard_size_hint.is_none()
            && max_shard_too_big(max_shard, num_keys, shard_edge.num_shards(), self.eps)
        {
            return Err(SolveError::MaxShardTooBig.into());
        }

        (self.c, self.aug) = shard_edge.set_up_graphs(num_keys, max_shard);
        self.num_keys = num_keys;

        let store = Box::new(shard_store) as Box<dyn ShardStore<S, V> + Send + Sync>;

        build_fn(self, seed, store, max_value, num_keys, pl, state)
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
    /// This is the core solver: it allocates the backend storage with
    /// `new_data(bit_width, len)`, where `len` is the number of vertices
    /// over all shards, peels the hypergraph defined by the shard iterator,
    /// writes the solution into the backend, and returns the assembled
    /// [`VFunc`].
    ///
    /// # Preconditions
    ///
    /// * `seed` must be the seed used during store population.
    ///
    /// * `new_data` must return zero-initialized storage.
    ///
    /// * `self.shard_edge`, `self.c`, `self.aug`, and `self.num_keys` must
    ///   be set up by the caller (typically by [`try_solve_once`]).
    ///
    /// * `get_val` must be deterministic.
    ///
    /// # Errors
    ///
    /// Returns [`SolveError::UnsolvableShard`] or
    /// [`SolveError::DuplicateLocalSignature`] if the system cannot be
    /// solved with the current seed.
    ///
    /// The solving algorithm is selected based on `self.aug` and
    /// `self.low_mem`; see the [`low_mem`] field documentation for the
    /// automatic selection heuristic.
    ///
    /// [`try_solve_once`]: Self::try_solve_once
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
        bit_width: usize,
        new_data: fn(usize, usize) -> D,
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
        self.num_threads = shard_edge.num_shards().min(self.max_num_threads).max(1);

        let mut data = new_data(
            bit_width,
            shard_edge.num_vertices() * shard_edge.num_shards(),
        );

        pl.info(format_args!(
            "Number of keys: {}; bit width: {}",
            self.num_keys,
            data.bit_width(),
        ));

        pl.info(format_args!(
            "{}; signatures: {}",
            shard_edge,
            core::any::type_name::<S>()
        ));

        pl.info(format_args!(
            "c: {}; overhead: {:+.3}%",
            self.c,
            100. * ((shard_edge.num_vertices() * shard_edge.num_shards()) as f64
                / (self.num_keys as f64)
                - 1.),
        ));

        // main_pl (shard counter) stays at info; per-shard detail at trace.
        // The info level must be restored on the error path too, or a retry
        // after an unsolvable shard would log the following attempts at
        // trace level, making them invisible.
        let mut main_pl = pl.concurrent();
        pl.log_level(log::Level::Trace);

        let solve_result = if self.aug {
            self.par_solve(
                seed,
                shard_iter,
                &mut data,
                0,
                |this, shard_index, shard, data, pl| {
                    this.aug_shard(shard_index, shard, data, get_val, inspect, pl)
                },
                &mut main_pl,
                pl,
            )
        } else if self.low_mem == Some(true)
            || self.low_mem.is_none() && self.num_threads > 3 && shard_edge.num_shards() > 2
        {
            self.par_solve(
                seed,
                shard_iter,
                &mut data,
                0,
                |this, shard_index, shard, data, pl| {
                    this.peel_by_sig_vals_low_mem(shard_index, shard, data, get_val, inspect, pl)
                },
                &mut main_pl,
                pl,
            )
        } else {
            self.par_solve(
                seed,
                shard_iter,
                &mut data,
                0,
                |this, shard_index, shard, data, pl| {
                    this.peel_by_sig_vals_high_mem(shard_index, shard, data, get_val, inspect, pl)
                },
                &mut main_pl,
                pl,
            )
        };

        pl.log_level(log::Level::Info);
        solve_result?;

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

// XorGraph, DoubleStack, FastStack, and the remove_edge!
// macro live in [crate::func::peeling] so they can be shared with
// [crate::func::CompVFunc]. VBuilder's peeler bodies are otherwise
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
        seed: u64,
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
            .start(format!("Solving shards ({} threads)...", self.num_threads));

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
                                if TypeId::of::<V>() == TypeId::of::<EmptyVal>() {
                                    // For filters, we fill the array with random data,
                                    // otherwise elements with signature 0 would have a
                                    // significantly higher probability of being false
                                    // positives. The fill must happen for empty shards,
                                    // too, or their backing would stay all zero, which
                                    // is exactly the degeneracy we are avoiding; the
                                    // seed is mixed with the shard index so different
                                    // shards get different backgrounds, and it is the
                                    // attempt seed, so retries are not repeatable.
                                    //
                                    // SAFETY: We work around the fact that [usize] does
                                    // not implement Fill.
                                    Mwc192::seed_from_u64(seed ^ mix64(shard_index as u64))
                                        .fill_bytes(unsafe {
                                            data.as_mut_slice().align_to_mut::<u8>().1
                                        });
                                }

                                if shard.is_empty() {
                                    continue;
                                }

                                if self.failed.load(Ordering::Relaxed) {
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

                                    // The second conjunct is always false on 32-bit
                                    // platforms, disabling the check
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
                                        // by the ShardEdge trait, so we can view the
                                        // elements of the vector as a slice of the
                                        // latter type (the layout of Vec itself is
                                        // not guaranteed to be independent of the
                                        // element type, so we do not transmute it).
                                        let shard_len = shard.len();
                                        let sort_shard = unsafe {
                                            std::slice::from_raw_parts_mut(
                                                shard.as_mut_ptr() as *mut E::SortSigVal<V>,
                                                shard_len,
                                            )
                                        };

                                        let builder = sort_shard.radix_sort_builder();
                                        if self.max_num_threads == 1 {
                                            builder
                                                .with_single_threaded_tuner()
                                                .with_parallel(false)
                                        } else {
                                            builder
                                        }
                                        .sort();

                                        // In-place deduplication, as in
                                        // Vec::dedup: the first element of each
                                        // run of equal elements is moved to the
                                        // front, and the vector is then truncated
                                        let mut dedup_len = usize::from(shard_len > 0);
                                        for i in 1..shard_len {
                                            if sort_shard[i] != sort_shard[dedup_len - 1] {
                                                sort_shard.swap(dedup_len, i);
                                                dedup_len += 1;
                                            }
                                        }
                                        shard.truncate(dedup_len);

                                        if TypeId::of::<V>() == TypeId::of::<EmptyVal>() {
                                            // Duplicate local signatures on
                                            // large filters can be simply
                                            // removed: they do not change the
                                            // semantics of the filter because
                                            // hashes are computed on
                                            // local signatures.
                                            pl.info(format_args!(
                                                "Removed signatures: {}",
                                                shard_len - dedup_len
                                            ));
                                        } else {
                                            // For function, we have to try again
                                            if shard_len != dedup_len {
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
    /// This peeler cannot be used in conjunction with [augmented peeling], as
    /// after a failed peeling it is not possible to retrieve information about
    /// the signature/value pairs in the shard.
    ///
    /// [`peel_by_sig_vals_low_mem`]: VBuilder::peel_by_sig_vals_low_mem
    /// [augmented peeling]: VBuilder::aug_shard
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
    /// This peeler cannot be used in conjunction with [augmented peeling], as
    /// after a failed peeling it is not possible to retrieve information about
    /// the signature/value pairs in the shard.
    ///
    /// [`peel_by_sig_vals_high_mem`]: VBuilder::peel_by_sig_vals_high_mem
    /// [augmented peeling]: VBuilder::aug_shard
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

    /// Solves a shard of given index using augmented peeling, and stores the
    /// solution in the given data.
    ///
    /// The graph is peeled by edge indices; whenever peeling gets stuck, we
    /// remove the first live edge in shard order and continue peeling. Since
    /// shards are sorted by the segment of the first vertex of their edges,
    /// the removed edge lies in the leftmost segment containing live edges,
    /// and peeling resumes from the hole left by the removal, as at the
    /// lightly loaded left end of a fuse graph. The procedure is a transposed
    /// form of the inactivation decoding of Raptor codes (see
    /// [`FuseAug3Shards`](crate::func::shard_edge::FuseAug3Shards)).
    ///
    /// At the end, every vertex that is incident to some edge but is not the
    /// pivot of a peeled edge is *lost*: its value can be chosen freely, and
    /// the values of the lost vertices determine, by back-substitution, the
    /// values of all pivots. Each removed edge is thus a linear equation in
    /// the lost vertices; moreover, only vertices lost after the first
    /// removal can appear in such equations, as removed edges depend only on
    /// the pivots of edges peeled after the first removal. We compute the
    /// coefficients of the equations for the most recently lost vertices, in
    /// blocks of 256 vertices, and solve the resulting small dense system,
    /// starting with twice as many unknowns as removed edges and doubling the
    /// number of blocks until the system is solvable or the lost vertices are
    /// exhausted.
    ///
    /// Without the cap on the number of unknowns (four times the number of
    /// removed edges, plus 2048) the solver would be exact, that is, it would
    /// fail only on unsolvable systems; the cap makes attempts on unsolvable
    /// systems fail fast.
    fn aug_shard<
        V: BinSafe,
        G: Fn(&E, SigVal<E::LocalSig, V>) -> D::Value + Send + Sync,
        H: Fn(&SigVal<S, V>) + Send + Sync,
    >(
        &self,
        shard_index: usize,
        shard: Arc<Vec<SigVal<S, V>>>,
        mut data: ShardData<'_, D>,
        get_val: &G,
        inspect: &H,
        pl: &mut impl ProgressLog,
    ) -> Result<(), ()> {
        const BLOCK_WORDS: usize = 4;
        let shard_edge = &self.shard_edge;
        let num_vertices = shard_edge.num_vertices();
        let num_shards = shard_edge.num_shards();
        let num_edges = shard.len();

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
        pl.done_with_count(num_edges);

        assert!(
            !xor_graph.overflow,
            "Degree overflow for shard {}/{}",
            shard_index + 1,
            num_shards
        );

        if self.failed.load(Ordering::Relaxed) {
            return Err(());
        }

        pl.start(format!(
            "Peeling graph for shard {}/{} by edge indices (augmented)...",
            shard_index + 1,
            num_shards
        ));

        let edge =
            |edge_index: usize| shard_edge.local_edge(shard_edge.local_sig(shard[edge_index].sig));
        let value = |edge_index: usize| {
            let sig_val = &shard[edge_index];
            get_val(
                shard_edge,
                SigVal {
                    sig: shard_edge.local_sig(sig_val.sig),
                    val: sig_val.val,
                },
            )
        };

        // The lower stack contains vertices to be visited. The upper stack
        // contains peeled edges.
        let mut double_stack = DoubleStack::<E::Vertex>::new(num_vertices);
        let mut sides_stack = Vec::<u8>::with_capacity(num_edges);
        let mut dead: BitVec = BitVec::new(num_edges);
        let mut removed = Vec::<E::Vertex>::new();
        let mut lost = Vec::<E::Vertex>::new();
        // Number of edges peeled before the first removal
        let mut num_peeled_before = 0;
        let mut next_removal = 0;

        for (v, degree) in xor_graph.degrees().enumerate() {
            if degree == 1 {
                double_stack.push_lower(E::Vertex::as_from(v));
            }
        }

        loop {
            while let Some(v) = double_stack.pop_lower() {
                let v: usize = v.as_to();
                if xor_graph.degree(v) == 0 {
                    continue;
                }
                let (edge_index, side) = xor_graph.edge_and_side(v);
                xor_graph.zero(v);
                double_stack.push_upper(edge_index);
                sides_stack.push(side as u8);
                let e_index: usize = edge_index.as_to();
                dead.set(e_index, true);
                let e = edge(e_index);
                for (s, &u) in e.iter().enumerate() {
                    if s == side {
                        continue;
                    }
                    match xor_graph.degree(u) {
                        2 => double_stack.push_lower(E::Vertex::as_from(u)),
                        1 if !removed.is_empty() => lost.push(E::Vertex::as_from(u)),
                        _ => {}
                    }
                    xor_graph.remove(u, edge_index, s);
                }
            }

            if double_stack.upper_len() + removed.len() == num_edges {
                break;
            }

            // Stuck: we remove the first live edge in shard order
            while dead[next_removal] {
                next_removal += 1;
            }
            if removed.is_empty() {
                num_peeled_before = double_stack.upper_len();
            }
            let edge_index = E::Vertex::as_from(next_removal);
            dead.set(next_removal, true);
            removed.push(edge_index);
            for (s, &u) in edge(next_removal).iter().enumerate() {
                match xor_graph.degree(u) {
                    2 => double_stack.push_lower(E::Vertex::as_from(u)),
                    1 => lost.push(E::Vertex::as_from(u)),
                    _ => {}
                }
                xor_graph.remove(u, edge_index, s);
            }
        }

        pl.done();
        drop(xor_graph);
        drop(dead);

        let num_removed = removed.len();

        // Assigns values to the pivots of the given peeled edges, which
        // must be in reverse peeling order
        let assign = |data: &mut ShardData<'_, D>, edges: &mut dyn Iterator<Item = (usize, u8)>| {
            for (edge_index, side) in edges {
                let e = edge(edge_index);
                let side = side as usize;
                // SAFETY: vertex indices from local_edge are guaranteed
                // within bounds of data by the ShardEdge contract; side is
                // always 0, 1, or 2.
                unsafe {
                    let xor = match side {
                        0 => data.get_value_unchecked(e[1]) ^ data.get_value_unchecked(e[2]),
                        1 => data.get_value_unchecked(e[0]) ^ data.get_value_unchecked(e[2]),
                        2 => data.get_value_unchecked(e[0]) ^ data.get_value_unchecked(e[1]),
                        _ => core::hint::unreachable_unchecked(),
                    };
                    data.set_value_unchecked(e[side], value(edge_index) ^ xor);
                }
            }
        };

        let num_peeled = double_stack.upper_len();
        let peeled = || {
            double_stack
                .iter_upper()
                .map(|&e| e.as_to())
                .zip(sides_stack.iter().rev().copied())
        };

        if num_removed != 0 {
            if self.failed.load(Ordering::Relaxed) {
                return Err(());
            }
            if lost.len() < num_removed {
                pl.debug(format_args!(
                    "Augmented peeling failed for shard {}/{}: {} removed edges but only {} lost vertices",
                    shard_index + 1,
                    num_shards,
                    num_removed,
                    lost.len()
                ));
                return Err(());
            }

            pl.start(format!(
                "Eliminating peeled variables for shard {}/{} ({} removed edges, {} lost vertices)...",
                shard_index + 1,
                num_shards,
                num_removed,
                lost.len()
            ));

            // Removed equations depend only on edges peeled after the first
            // removal (the first num_peeled - num_peeled_before in reverse
            // peeling order). We assign their pivots using the current
            // values of the lost vertices (zero), and evaluate the removed
            // equations to get the constant terms.
            let num_tail = num_peeled - num_peeled_before;
            assign(&mut data, &mut peeled().take(num_tail));
            let rhs: Vec<D::Value> = removed
                .iter()
                .map(|&r| {
                    let r: usize = r.as_to();
                    let e = edge(r);
                    // SAFETY: see above
                    unsafe {
                        value(r)
                            ^ data.get_value_unchecked(e[0])
                            ^ data.get_value_unchecked(e[1])
                            ^ data.get_value_unchecked(e[2])
                    }
                })
                .collect();
            // Pivot first, then the two other vertices
            let tail: Vec<[E::Vertex; 3]> = peeled()
                .take(num_tail)
                .map(|(edge_index, side)| {
                    let e = edge(edge_index);
                    let (a, b) = match side {
                        0 => (e[1], e[2]),
                        1 => (e[0], e[2]),
                        _ => (e[0], e[1]),
                    };
                    [
                        E::Vertex::as_from(e[side as usize]),
                        E::Vertex::as_from(a),
                        E::Vertex::as_from(b),
                    ]
                })
                .collect();
            let removed_edges: Vec<[usize; 3]> = removed.iter().map(|&r| edge(r.as_to())).collect();
            drop(removed);

            // Unknown j is the j-th most recently lost vertex. We cap the
            // number of unknowns so that an unsolvable system fails fast;
            // solvable systems need on average between one and one and a
            // half unknowns per removed edge, and occasionally a few more
            // when few edges have been removed.
            let num_lost = lost.len().min(4 * num_removed + 2048);
            let lost = &lost[lost.len() - num_lost..];
            let unknown = |j: usize| -> usize { lost[num_lost - 1 - j].as_to() };
            let block_size = BLOCK_WORDS * 64;
            let max_blocks = num_lost.div_ceil(block_size);
            let mut coeffs = vec![[0_u64; BLOCK_WORDS]; num_vertices];
            // Coefficients of the removed equations, block by block
            let mut blocks: Vec<Vec<[u64; BLOCK_WORDS]>> = vec![];
            let mut num_blocks = (2 * num_removed).div_ceil(block_size).min(max_blocks);
            let solution = loop {
                while blocks.len() < num_blocks {
                    let base = blocks.len() * block_size;
                    let top = (base + block_size).min(num_lost);
                    coeffs.iter_mut().for_each(|c| *c = [0; BLOCK_WORDS]);
                    for j in base..top {
                        let o = j - base;
                        coeffs[unknown(j)][o / 64] |= 1 << (o % 64);
                    }
                    for &[p, a, b] in &tail {
                        let (p, a, b): (usize, usize, usize) = (p.as_to(), a.as_to(), b.as_to());
                        let (ca, cb) = (coeffs[a], coeffs[b]);
                        coeffs[p] = std::array::from_fn(|i| ca[i] ^ cb[i]);
                    }
                    blocks.push(
                        removed_edges
                            .iter()
                            .map(|e| {
                                let (c0, c1, c2) = (coeffs[e[0]], coeffs[e[1]], coeffs[e[2]]);
                                std::array::from_fn(|i| c0[i] ^ c1[i] ^ c2[i])
                            })
                            .collect(),
                    );
                }

                let row_words = blocks.len() * BLOCK_WORDS;
                let mut rows = vec![0_u64; num_removed * row_words];
                for (b, block) in blocks.iter().enumerate() {
                    for (i, c) in block.iter().enumerate() {
                        let start = i * row_words + b * BLOCK_WORDS;
                        rows[start..start + BLOCK_WORDS].copy_from_slice(c);
                    }
                }
                let mut block_rhs = rhs.clone();
                let num_vars = (blocks.len() * block_size).min(num_lost);
                match crate::utils::mod2_sys::dense_gaussian_elimination(
                    &mut rows,
                    &mut block_rhs,
                    num_removed,
                    row_words,
                    num_vars,
                ) {
                    Some(solution) => break solution,
                    None if blocks.len() == max_blocks => {
                        pl.debug(format_args!(
                            "Augmented peeling failed for shard {}/{}: unsolvable system",
                            shard_index + 1,
                            num_shards,
                        ));
                        return Err(());
                    }
                    None => num_blocks = (2 * blocks.len()).min(max_blocks),
                }
            };
            pl.done_with_count(num_removed);

            // The solution is the delta to apply to the lost vertices
            for (j, &x) in solution.iter().enumerate() {
                let v = unknown(j);
                // SAFETY: lost vertices are within bounds of data
                unsafe {
                    data.set_value_unchecked(v, data.get_value_unchecked(v) ^ x);
                }
            }
        }

        if self.failed.load(Ordering::Relaxed) {
            return Err(());
        }

        pl.start(format!(
            "Assigning values for shard {}/{}...",
            shard_index + 1,
            num_shards
        ));
        assign(&mut data, &mut peeled());
        pl.done();

        Ok(())
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
            // SAFETY: vertex indices from local_edge are guaranteed within
            // bounds of data by the ShardEdge contract; side is always
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

#[cfg(test)]
mod retry_state_tests {
    use super::*;

    fn retry_state() -> RetryState {
        RetryState {
            prng: SmallRng::seed_from_u64(0),
            dup_count: 0,
            local_dup_count: 0,
            max_shard_too_big: 0,
            unsolvable_count: 0,
        }
    }

    /// The `MaxShardTooBig` retry counter is capped: the first 99 occurrences
    /// are retried, the 100th is fatal, so a pathological configuration cannot
    /// loop forever. The cap does not depend on augmented peeling.
    #[test]
    fn max_shard_too_big_is_bounded() {
        for aug in [false, true] {
            let mut pl = no_logging![];
            let mut rs = retry_state();
            for i in 0..99 {
                let r = rs.handle_solve_result(
                    Err::<(), _>(SolveError::MaxShardTooBig.into()),
                    aug,
                    &mut pl,
                );
                assert!(matches!(r, Ok(None)), "attempt {i} should be retried");
            }
            let r = rs.handle_solve_result(
                Err::<(), _>(SolveError::MaxShardTooBig.into()),
                aug,
                &mut pl,
            );
            assert!(
                matches!(
                    r.unwrap_err().downcast_ref::<BuildError>(),
                    Some(BuildError::MaxShardTooBig)
                ),
                "the 100th MaxShardTooBig must be fatal"
            );
        }
    }

    /// The `UnsolvableShard` retry counter is capped, too, and the 100th
    /// failure (the 1000th with augmented peeling) returns a typed
    /// error instead of panicking.
    #[test]
    fn unsolvable_shard_is_bounded() {
        for (aug, max_attempts) in [
            (false, RetryState::MAX_ATTEMPTS),
            (true, RetryState::MAX_AUG_ATTEMPTS),
        ] {
            let mut pl = no_logging![];
            let mut rs = retry_state();
            for i in 0..max_attempts - 1 {
                let r = rs.handle_solve_result(
                    Err::<(), _>(SolveError::UnsolvableShard.into()),
                    aug,
                    &mut pl,
                );
                assert!(matches!(r, Ok(None)), "attempt {i} should be retried");
            }
            let r = rs.handle_solve_result(
                Err::<(), _>(SolveError::UnsolvableShard.into()),
                aug,
                &mut pl,
            );
            assert!(
                matches!(
                    r.unwrap_err().downcast_ref::<BuildError>(),
                    Some(BuildError::UnsolvableShard)
                ),
                "the {max_attempts}th UnsolvableShard must be fatal"
            );
        }
    }

    /// The shard-size acceptance threshold is the maximum of the ε-cost
    /// term and a statistical term covering the natural fluctuation of
    /// the maximum shard size.
    #[test]
    fn max_shard_too_big_scales_with_eps() {
        // Fuse-like regime: 10⁹ keys in 100 shards, average 10⁷. The
        // ε-cost term dominates: tolerance = 5·0.001·10⁷ = 50000.
        assert!(!max_shard_too_big(10_050_000, 1_000_000_000, 100, 0.001));
        assert!(max_shard_too_big(10_060_000, 1_000_000_000, 100, 0.001));
        // Small shards: 800000 keys in 16 shards, average 50000. The
        // statistical term dominates: tolerance = 2.5·√(50000·ln 16) ≈
        // 930, so a typical maximum (≈ average + 400) is accepted—the
        // old 0.5% rule rejected it about 60 times per build—while a
        // pathological 4% imbalance is still rejected.
        assert!(!max_shard_too_big(50_500, 800_000, 16, 0.001));
        assert!(max_shard_too_big(52_000, 800_000, 16, 0.001));
    }
}
