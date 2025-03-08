/*
 * SPDX-FileCopyrightText: 2025 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]

use crate::func::{*, shard_edge::ShardEdge};
use crate::bits::*;
use crate::dict::VFilter;
use crate::traits::bit_field_slice::{BitFieldSlice, BitFieldSliceMut, Word};
use crate::utils::*;
use common_traits::CastableInto;
use derivative::Derivative;
use derive_setters::*;
use dsi_progress_logger::*;
use epserde::prelude::*;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand::{Rng, RngCore};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use rdst::*;
use std::any::TypeId;
use std::hint::unreachable_unchecked;
use std::marker::PhantomData;
use std::ops::{BitXor, BitXorAssign};
use std::slice::Iter;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use super::shard_edge::FuseLge3Shards;

const LOG2_MAX_SHARDS: u32 = 12;

/// A builder for [`VFunc`] and [`VFilter`].
///
/// Keys must implement the [`ToSig`] trait, which provides a method to compute
/// a signature of the key.
///
/// There are two construction modes: in core memory (default) and
/// [offline](VBuilder::offline). In the first case, space will be allocated in
/// core memory for signatures and associated values for all keys; in the second
/// case, such information will be stored in a number of on-disk buckets using a
/// [`SigStore`]. It is also possible to [set the maximum number of
/// threads](VBuilder::max_num_threads).
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
/// All construction methods require to pass one or two [`RewindableIoLender`]s
/// (keys and possibly values), and the construction might fail and keys might
/// be scanned again. The structures in the [`lenders`] modules module provide
/// easy ways to build such lenders, even starting from compressed files of
/// UTF-8 strings. The type of the keys of the resulting filter or function will
/// be the type of the elements of the [`RewindableIoLender`].
///
/// # Examples
///
/// In this example, we build a function that maps each key to itself. Note that
/// setter for the expected number of keys is used to optimize the construction.
/// Note that we use the [`FromIntoIterator`] adapter to turn a clonable
/// [`IntoIterator`] into a [`RewindableIoLender`].
///
/// ```rust
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use sux::func::vbuilder::VBuilder;
/// use dsi_progress_logger::no_logging;
/// use sux::utils::FromIntoIterator;
///
/// let builder = VBuilder::<usize, Box<[usize]>>::default()
///     .expected_num_keys(100);
/// let func = builder.try_build_func(
///    FromIntoIterator::from(0..100),
///    FromIntoIterator::from(0..100),
///    no_logging![]
/// )?;
///
/// for i in 0..100 {
///    assert_eq!(i, func.get(&i));
/// }
/// #     Ok(())
/// # }
/// ```
///
/// Alternatively we can use the bit-field vector backend, that will use
/// ⌈log₂(99)⌉ bits per element:
///
/// ```rust
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use sux::func::vbuilder::VBuilder;
/// use dsi_progress_logger::no_logging;
/// use sux::utils::FromIntoIterator;
/// use sux::bits::BitFieldVec;
///
/// let builder = VBuilder::<usize, BitFieldVec<usize>>::default()
///     .expected_num_keys(100);
/// let func = builder.try_build_func(
///    FromIntoIterator::from(0..100),
///    FromIntoIterator::from(0..100),
///    no_logging![]
/// )?;
///
/// for i in 0..100 {
///    assert_eq!(i, func.get(&i));
/// }
/// #     Ok(())
/// # }
/// ```
///
/// We now try to build a filter for the same key set:
///
/// ```rust
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use sux::func::vbuilder::VBuilder;
/// use dsi_progress_logger::no_logging;
/// use sux::utils::FromIntoIterator;
///
/// let builder = VBuilder::<usize, Box<[usize]>>::default()
///     .expected_num_keys(100);
/// let func = builder.try_build_filter(
///    FromIntoIterator::from(0..100),
///    no_logging![]
/// )?;
///
/// for i in 0..100 {
///    assert!(func[i]);
/// }
/// #     Ok(())
/// # }

#[derive(Setters, Debug, Derivative)]
#[derivative(Default)]
#[setters(generate = false)]
pub struct VBuilder<
    W: ZeroCopy + Word,
    D: BitFieldSlice<W> + Send + Sync = Box<[W]>,
    S = [u64; 2],
    E: ShardEdge<S, 3> = FuseLge3Shards,
> {
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
    max_num_threads: usize,

    /// Use disk-based buckets to reduce core memory usage at construction time.
    #[setters(generate = true)]
    offline: bool,

    /// Check for duplicated signatures. This is not necessary in general,
    /// but if you suspect you might be feeding duplicate keys, you can
    /// enable this check.
    #[setters(generate = true)]
    check_dups: bool,

    /// The seed for the random number generator.
    #[setters(generate = true)]
    seed: u64,

    /// The base-2 logarithm of buckets of the [`SigStore`]. The default is 8.
    /// This value is automatically overriden, even if set, if you provide an
    /// [expected number of keys](VBuilder::expected_num_keys).
    #[setters(generate = true, strip_option)]
    #[derivative(Default(value = "8"))]
    log2_buckets: u32,

    /// The bit width of the maximum value.
    bit_width: usize,
    /// The edge generator.
    shard_edge: E,
    /// The number of keys.
    num_keys: usize,
    /// The ratio between the number of variables and the number of equations.
    c: f64,
    /// Whether we should use lazy Gaussian elimination.
    lge: bool,
    /// Fast-stop for failed attempts.
    failed: AtomicBool,
    #[doc(hidden)]
    _marker_v: PhantomData<(W, D, S)>,
}

/// Fatal build errors.
#[derive(thiserror::Error, Debug)]
pub enum BuildError {
    #[error("Duplicate key")]
    /// A duplicate key was detected.
    DuplicateKey,
}

#[derive(thiserror::Error, Debug)]
/// Transient error during the build, leading to
/// trying with a different seed.
pub enum SolveError {
    #[error("Duplicate signature")]
    /// A duplicate signature was detected.
    DuplicateSignature,
    #[error("Max shard too big")]
    /// The maximum shard is too big.
    MaxShardTooBig,
    #[error("Unsolvable shard")]
    /// A shard cannot be solved.
    UnsolvableShard,
}

/// The result of a peeling operation.
enum PeelResult<
    'a,
    W: ZeroCopy + Word + Send + Sync,
    D: BitFieldSlice<W> + BitFieldSliceMut<W> + Send + Sync + 'a,
    S: Sig + ZeroCopy + Send + Sync,
    V: ZeroCopy,
> {
    Complete(),
    Partial {
        /// The shard index.
        shard_index: usize,
        /// The shard.
        shard: &'a [SigVal<S, V>],
        /// The data.
        data: ShardData<'a, W, D>,
        /// The double stack whose upper stack contains the peeled edges.
        double_stack: DoubleStack<u32>,
        /// The sides stack.
        sides_stack: Vec<u8>,
    },
}

/// A graph represented compactly.
///
/// Each (*k*-hyper)edge is a set of *k* vertices (by construction fuse graphs
/// to not have degenerate edges), but we represent it internally as a vector.
/// We call *side* the position of a vertex in the edge.
///
/// For each vertex, information about the edges incident to the vertex and the
/// sides of the vertex in such edges. While technically not necessary to
/// perform peeling, the knowledge of the sides speeds up the peeling visit by
/// reducing the number of tests that are necessary to update the degrees once
/// the edge is peeled (see the `peel_shard` method). For the same reason it
/// also speeds up assignment.
///
/// Depending on the peeling method, the graph will store edge indices or
/// signature/value pairs (the generic parameter `X`).
///
/// Edge information is packed together using Djamal's XOR trick (see
/// [“Cache-Oblivious Peeling of Random
/// Hypergraphs”](https://doi.org/10.1109/DCC.2014.48)): since during the
/// peeling b visit we need to know the content of the list only when a single
/// edge index is present, we can XOR together all the edge information.
///
/// Assuming less than 2³² vertices in a shard, we can store XOR'd edges in a
/// `u32`. We then use a single byte to store the degree (six upper bits) and
/// the XOR of the sides (lower two bits). The degree can be stored with a small
/// number of bits because the graph is random, so the maximum degree is *O*(log
/// log *n*).
///
/// When we peel an edge, we just zero the degree, leaving the edge information
/// and the side in place for further processing later.
///
/// This approach reduces the core memory usage for the hypergraph to 5 bytes
/// per vertex when storing edge indices. Edges are derived on the fly from
/// signatures using the edge indices and indexing the shard. If instead
/// signature/value pairs are stored, the memory usage is significantly higher,
/// but obtaining an edge does not require accessing the shard.
struct XorGraph<X: BitXor + BitXorAssign + Default + Copy> {
    edges: Box<[X]>,
    degrees_sides: Box<[u8]>,
}

impl<X: BitXor + BitXorAssign + Default + Copy> XorGraph<X> {
    pub fn new(n: usize) -> XorGraph<X> {
        XorGraph {
            edges: vec![X::default(); n].into(),
            degrees_sides: vec![0; n].into(),
        }
    }

    #[inline(always)]
    pub fn add(&mut self, v: usize, x: X, side: usize) {
        debug_assert!(side < 3);
        self.degrees_sides[v] += 4;
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
    pub fn edge_index_and_side(&self, v: usize) -> (X, usize) {
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

/// Two stacks in the same vector.
///
/// This struct implements a pair of stacks sharing the same memory. The lower
/// stack grows from the beginning of the vector, the upper stack grows from the
/// end of the vector. Since we use the lower stack for peeled vertices and the
/// upper stack for vertices to visit, the sum of the lengths of the two stacks
/// cannot exceed the length of the vector.
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
type ShardDataIter<'a, W, D> = <D as BitFieldSliceMut<W>>::ChunksMut<'a>;
/// A segment of data associated with a specific shard.
type ShardData<'a, W, D> = <ShardDataIter<'a, W, D> as Iterator>::Item;

/// Builds a new function using a `Box<[W]>` to store values.
///
/// Since values are stored in a slice, access is particularly fast, but the bit
/// width of the output of the function will be  exactly the bit width of `W`.
impl<W: ZeroCopy + Word, S: Sig + Send + Sync, E: ShardEdge<S, 3>> VBuilder<W, Box<[W]>, S, E>
where
    SigVal<S, W>: RadixKey + BitXor + BitXorAssign,
    Box<[W]>: BitFieldSliceMut<W> + BitFieldSlice<W>,
{
    pub fn try_build_func<T: ?Sized + ToSig<S> + std::fmt::Debug>(
        mut self,
        keys: impl RewindableIoLender<T>,
        values: impl RewindableIoLender<W>,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> anyhow::Result<VFunc<T, W, Box<[W]>, S, E>>
    where
        for<'a> ShardDataIter<'a, W, Box<[W]>>: Send,
        for<'a> ShardData<'a, W, Box<[W]>>: Send,
    {
        let get_val = |sig_val: &SigVal<S, W>| sig_val.val;
        let new_data = |_bit_width: usize, len: usize| vec![W::ZERO; len].into();
        self.build_loop(keys, values, &get_val, new_data, pl)
    }
}

/// Builds a new filter using a `Box<[W]>` to store values.
///
/// Since values are stored in a slice access is particularly fast, but the
/// number of signature bits will be  exactly the bit width of `W`.
impl<W: ZeroCopy + Word, S: Sig + Send + Sync, E: ShardEdge<S, 3>> VBuilder<W, Box<[W]>, S, E>
where
    SigVal<S, EmptyVal>: RadixKey + BitXor + BitXorAssign,
    Box<[W]>: BitFieldSliceMut<W> + BitFieldSlice<W>,
    u64: CastableInto<W>,
{
    pub fn try_build_filter<T: ?Sized + ToSig<S> + std::fmt::Debug>(
        mut self,
        keys: impl RewindableIoLender<T>,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> anyhow::Result<VFilter<W, VFunc<T, W, Box<[W]>, S, E>>>
    where
        for<'a> ShardDataIter<'a, W, Box<[W]>>: Send,
        for<'a> ShardData<'a, W, Box<[W]>>: Send,
    {
        let filter_mask = W::MAX;
        let get_val = |sig_val: &SigVal<S, EmptyVal>| sig_val.sig.sig_u64().cast();
        let new_data = |_bit_width: usize, len: usize| vec![W::ZERO; len].into();

        Ok(VFilter {
            func: self.build_loop(
                keys,
                FromIntoIterator::from(itertools::repeat_n(EmptyVal::default(), usize::MAX)),
                &get_val,
                new_data,
                pl,
            )?,
            filter_mask,
            sig_bits: W::BITS as u32,
        })
    }
}

/// Builds a new function using a [bit-field vector](BitFieldVec) on words of
/// type `W` to store values.
///
/// Since values are stored in a bit-field vector, access will be slower than
/// when using a boxed slice, but the bit width of stored values will be the
/// minimum necessary. It must be in any case at most the bit width of `W`.
///
/// Typically `W` will be `usize` or `u64`. You can use `u128` if the bit width
/// of the values is larger than 64.
impl<W: ZeroCopy + Word, S: Sig + Send + Sync, E: ShardEdge<S, 3>> VBuilder<W, BitFieldVec<W>, S, E>
where
    SigVal<S, W>: RadixKey + BitXor + BitXorAssign,
{
    pub fn try_build_func<T: ?Sized + ToSig<S> + std::fmt::Debug>(
        mut self,
        keys: impl RewindableIoLender<T>,
        values: impl RewindableIoLender<W>,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> anyhow::Result<VFunc<T, W, BitFieldVec<W>, S, E>> {
        let get_val = |sig_val: &SigVal<S, W>| sig_val.val;
        let new_data = |bit_width, len| BitFieldVec::<W>::new(bit_width, len);
        self.build_loop(keys, values, &get_val, new_data, pl)
    }
}

/// Builds a new filter using a [bit-field vector](BitFieldVec) on words of type
/// `W` to store values.
///
/// Since values are stored in a bit-field vector, access will be slower than
/// when using a boxed slice, but the signature bits can be set at will. They
/// must be in any case at most the bit width of `W`.
///
/// Typically `W` will be `usize` or `u64`.
impl<W: ZeroCopy + Word, S: Sig + Send + Sync, E: ShardEdge<S, 3>> VBuilder<W, BitFieldVec<W>, S, E>
where
    SigVal<S, EmptyVal>: RadixKey + BitXor + BitXorAssign,
    u64: CastableInto<W>,
{
    pub fn try_build_filter<T: ?Sized + ToSig<S> + std::fmt::Debug>(
        mut self,
        keys: impl RewindableIoLender<T>,
        filter_bits: u32,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> anyhow::Result<VFilter<W, VFunc<T, W, BitFieldVec<W>, S, E>>> {
        assert!(filter_bits > 0);
        assert!(filter_bits <= W::BITS as u32);
        let filter_mask = W::MAX >> (W::BITS as u32 - filter_bits);
        let get_val = |sig_val: &SigVal<S, EmptyVal>| sig_val.sig.sig_u64().cast() & filter_mask;
        let new_data = |bit_width, len| BitFieldVec::<W>::new(bit_width, len);

        Ok(VFilter {
            func: self.build_loop(
                keys,
                FromIntoIterator::from(itertools::repeat_n(EmptyVal::default(), usize::MAX)),
                &get_val,
                new_data,
                pl,
            )?,
            filter_mask,
            sig_bits: filter_bits,
        })
    }
}

impl<
        W: ZeroCopy + Word,
        D: BitFieldSlice<W> + BitFieldSliceMut<W> + Send + Sync,
        S: Sig + Send + Sync,
        E: ShardEdge<S, 3>,
    > VBuilder<W, D, S, E>
{
    /// Builds and return a new function with given keys and values.
    ///
    /// This function can build functions based both on vectors and on bit-field
    /// vectors. The necessary abstraction is provided by the `new_data(bit_width,
    /// len)` function, which is called to create the data structure to store
    /// the values.
    ///
    /// When `V` is [`EmptyVal`], the this method builds a function supporting a
    /// filter by mapping each key to its signature. The necessary abstraction
    /// is provided by the `get_val` function, which is called to extract the
    /// value from the signature/value pair`; in the case of functions it
    /// returns the value stored in the signature/value pair, and in the case of
    /// filters it returns the lower bits of [`SigVal::sig_u64`].
    fn build_loop<T: ?Sized + ToSig<S> + std::fmt::Debug, V: ZeroCopy + Default + Send + Sync>(
        &mut self,
        mut keys: impl RewindableIoLender<T>,
        mut values: impl RewindableIoLender<V>,
        get_val: &(impl Fn(&SigVal<S, V>) -> W + Send + Sync),
        new_data: fn(usize, usize) -> D,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> anyhow::Result<VFunc<T, W, D, S, E>>
    where
        SigVal<S, V>: RadixKey + BitXor + BitXorAssign + Send + Sync,
        for<'a> ShardDataIter<'a, W, D>: Send,
        for<'a> <ShardDataIter<'a, W, D> as Iterator>::Item: Send,
    {
        let mut dup_count = 0;
        let mut prng = SmallRng::seed_from_u64(self.seed);

        if let Some(expected_num_keys) = self.expected_num_keys {
            self.shard_edge.set_up_shards(expected_num_keys);
            self.log2_buckets = self.shard_edge.shard_high_bits();
        }

        pl.info(format_args!("Using 2^{} buckets", self.log2_buckets));

        // Loop until success or duplicate detection
        loop {
            let seed = prng.random();
            pl.expected_updates(self.expected_num_keys);
            pl.item_name("key");
            pl.start(format!(
                "Reading input and hashing keys to {} bits...",
                std::mem::size_of::<S>() * 8
            ));

            values = values.rewind()?;
            keys = keys.rewind()?;

            match if self.offline {
                self.try_seed(
                    seed,
                    sig_store::new_offline::<S, V>(
                        self.log2_buckets,
                        LOG2_MAX_SHARDS,
                        self.expected_num_keys,
                    )?,
                    &mut keys,
                    &mut values,
                    get_val,
                    new_data,
                    pl,
                )
            } else {
                self.try_seed(
                    seed,
                    sig_store::new_online::<S, V>(
                        self.log2_buckets,
                        LOG2_MAX_SHARDS,
                        self.expected_num_keys,
                    )?,
                    &mut keys,
                    &mut values,
                    get_val,
                    new_data,
                    pl,
                )
            } {
                Ok(func) => {
                    return Ok(func);
                }
                Err(error) => {
                    match error.downcast::<SolveError>() {
                        Ok(vfunc_error) => match vfunc_error {
                            // Let's try another seed, but just a few times--most likely,
                            // duplicate keys
                            SolveError::DuplicateSignature => {
                                if dup_count >= 3 {
                                    pl.error(format_args!("Duplicate keys (duplicate 128-bit signatures with four different seeds)"));
                                    return Err(BuildError::DuplicateKey.into());
                                }
                                pl.warn(format_args!(
                                "Duplicate 128-bit signature, trying again with a different seed..."
                            ));
                                dup_count += 1;
                            }
                            SolveError::MaxShardTooBig => {
                                pl.warn(format_args!(
                                "The maximum shard is too big, trying again with a different seed..."
                               ));
                            }
                            // Let's just try another seed
                            SolveError::UnsolvableShard => {
                                pl.warn(format_args!(
                                    "Unsolvable shard, trying again with a different seed..."
                                ));
                            }
                        },
                        Err(error) => return Err(error),
                    }
                }
            }
        }
    }
}

impl<
        W: ZeroCopy + Word,
        D: BitFieldSlice<W> + BitFieldSliceMut<W> + Send + Sync,
        S: Sig + Send + Sync,
        E: ShardEdge<S, 3>,
    > VBuilder<W, D, S, E>
{
    /// Tries to build a function using specific seed. See the comments in the
    /// [`VBuilder::build_loop`] method for more details.
    ///
    /// This methods reads the input, sets up the shards, allocates the backend
    /// using `new_data`, and passes the backend and an iterator on shards to
    /// the [`VBuilder::try_build_from_shard_iter`] method.
    fn try_seed<
        T: ?Sized + ToSig<S> + std::fmt::Debug,
        V: ZeroCopy + Default + Send + Sync,
        G: Fn(&SigVal<S, V>) -> W + Send + Sync,
    >(
        &mut self,
        seed: u64,
        mut sig_store: impl SigStore<S, V>,
        keys: &mut impl RewindableIoLender<T>,
        values: &mut impl RewindableIoLender<V>,
        get_val: &G,
        new_data: fn(usize, usize) -> D,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> anyhow::Result<VFunc<T, W, D, S, E>>
    where
        SigVal<S, V>: RadixKey + BitXor + BitXorAssign,
        for<'a> ShardDataIter<'a, W, D>: Send,
        for<'a> <ShardDataIter<'a, W, D> as Iterator>::Item: Send,
    {
        let mut max_value = W::ZERO;

        while let Some(result) = keys.next() {
            match result {
                Ok(key) => {
                    pl.light_update();
                    // This might be an actual value, if we are building a
                    // function, or EmptyVal, if we are building a filter.
                    let &maybe_val = values.next().expect("Not enough values")?;
                    let sig_val = SigVal {
                        sig: T::to_sig(key, seed),
                        val: maybe_val,
                    };
                    let val = get_val(&sig_val);
                    max_value = Ord::max(max_value, val);
                    sig_store.try_push(sig_val)?;
                }
                Err(e) => return Err(e.into()),
            }
        }
        pl.done();

        let start = Instant::now();

        self.num_keys = sig_store.len();
        self.bit_width = max_value.len() as usize;

        let mut shard_store = sig_store.into_shard_store(self.shard_edge.shard_high_bits())?;
        let max_shard = shard_store.shard_sizes().iter().copied().max().unwrap_or(0);

        self.shard_edge.set_up_shards(self.num_keys);
        (self.c, self.lge) = self.shard_edge.set_up_graphs(self.num_keys, max_shard);

        pl.info(format_args!(
            "Number of keys: {} Max value: {} Bit width: {}",
            self.num_keys, max_value, self.bit_width,
        ));

        if self.shard_edge.shard_high_bits() != 0 {
            pl.info(format_args!(
                "Max shard / average shard: {:.2}%",
                (100.0 * max_shard as f64)
                    / (self.num_keys as f64 / self.shard_edge.num_shards() as f64)
            ));
        }

        if max_shard as f64 > 1.01 * self.num_keys as f64 / self.shard_edge.num_shards() as f64 {
            // This might sometimes happen with small sharded graphs
            Err(SolveError::MaxShardTooBig.into())
        } else {
            let data = new_data(
                self.bit_width,
                self.shard_edge.num_vertices() * self.shard_edge.num_shards(),
            );
            self.try_build_from_shard_iter(seed, data, shard_store.iter(), get_val, pl)
                .inspect(|_| {
                    pl.info(format_args!(
                        "Construction from hashes completed in {:.3} seconds ({} keys, {:.3} ns/key)",
                        start.elapsed().as_secs_f64(),
                        self.num_keys,
                        start.elapsed().as_nanos() as f64 / self.num_keys as f64
                    ));
                })
                .map_err(Into::into)
        }
    }

    /// Builds and return a new function starting from an iterator on shards.
    ///
    /// This method provide construction logic that is independent from the
    /// actual storage of the values (offline or in core memory.)
    ///
    /// See [`VBuilder::build_loop`] for more details on the parameters.
    fn try_build_from_shard_iter<
        T: ?Sized + ToSig<S>,
        I,
        P,
        V: ZeroCopy + Default + Send + Sync,
        G: Fn(&SigVal<S, V>) -> W + Send + Sync,
    >(
        &mut self,
        seed: u64,
        mut data: D,
        shard_iter: I,
        get_val: &G,
        pl: &mut P,
    ) -> Result<VFunc<T, W, D, S, E>, SolveError>
    where
        SigVal<S, V>: RadixKey + BitXor + BitXorAssign,
        P: ProgressLog + Clone + Send + Sync,
        I: Iterator<Item = Arc<Vec<SigVal<S, V>>>> + Send,
        for<'a> ShardDataIter<'a, W, D>: Send,
        for<'a> std::iter::Enumerate<
            std::iter::Zip<<I as IntoIterator>::IntoIter, ShardDataIter<'a, W, D>>,
        >: Send,
        for<'a> (
            usize,
            (
                Arc<Vec<SigVal<S, V>>>,
                <ShardDataIter<'a, W, D> as Iterator>::Item,
            ),
        ): Send,
    {
        let thread_pool = ThreadPoolBuilder::new()
            .num_threads(self.shard_edge.num_shards().min(self.max_num_threads))
            .build()
            .unwrap(); // Seriously, it's not going to fail

        pl.info(format_args!("{}", self.shard_edge));
        pl.info(format_args!(
            "c: {}, Overhead: {:.4}% Number of threads: {}",
            self.c,
            (self.shard_edge.num_vertices() * self.shard_edge.num_shards()) as f64
                / (self.num_keys as f64),
            thread_pool.current_num_threads()
        ));

        if self.lge {
            pl.info(format_args!("Peeling towards lazy Gaussian elimination"));
            self.par_solve(
                shard_iter,
                &mut data,
                |this, shard_index, shard, data, pl| {
                    this.lge_shard(shard_index, shard, data, get_val, pl)
                },
                &thread_pool,
                &mut pl.concurrent(),
                pl,
            )?;
        } else if self.shard_edge.num_shards() <= 2 {
            // More memory, but more speed
            self.par_solve(
                shard_iter,
                &mut data,
                |this, shard_index, shard, data, pl| {
                    this.peel_shard_by_sig_vals(shard_index, shard, data, get_val, pl)
                },
                &thread_pool,
                &mut pl.concurrent(),
                pl,
            )?;
        } else {
            // Much less memory, but slower
            self.par_solve(
                shard_iter,
                &mut data,
                |this, shard_index, shard, data, pl| {
                    this.peel_shard_by_edge_indices(shard_index, shard, data, get_val, pl)
                        .map(|_| ())
                        .map_err(|_| ())
                },
                &thread_pool,
                &mut pl.concurrent(),
                pl,
            )?;
        }

        pl.info(format_args!(
            "Bits/keys: {} ({:.2}%)",
            data.len() as f64 * self.bit_width as f64 / self.num_keys as f64,
            100.0 * data.len() as f64 / self.num_keys as f64,
        ));

        Ok(VFunc {
            seed,
            shard_edge: self.shard_edge,
            num_keys: self.num_keys,
            data,
            _marker_t: std::marker::PhantomData,
            _marker_w: std::marker::PhantomData,
            _marker_s: std::marker::PhantomData,
        })
    }
}

impl<
        W: ZeroCopy + Word + Send + Sync,
        D: BitFieldSlice<W> + BitFieldSliceMut<W> + Send + Sync,
        S: Sig + ZeroCopy + Send + Sync,
        E: ShardEdge<S, 3>,
    > VBuilder<W, D, S, E>
{
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
        V: ZeroCopy + Send + Sync,
        I: IntoIterator<Item = Arc<Vec<SigVal<S, V>>>> + Send,
        C: ConcurrentProgressLog + Send + Sync,
        P: ProgressLog + Clone + Send + Sync,
    >(
        &mut self,
        shard_iter: I,
        data: &'b mut D,
        solve_shard: impl Fn(&Self, usize, &'b mut [SigVal<S, V>], ShardData<'b, W, D>, &mut P) -> Result<(), ()>
            + Send
            + Sync,
        thread_pool: &rayon::ThreadPool,
        main_pl: &mut C,
        pl: &mut P,
    ) -> Result<(), SolveError>
    where
        I::IntoIter: Send,
        SigVal<S, V>: RadixKey + Send + Sync,
        for<'a> ShardDataIter<'a, W, D>: Send,
        for<'a> std::iter::Enumerate<
            std::iter::Zip<<I as IntoIterator>::IntoIter, ShardDataIter<'a, W, D>>,
        >: Send,
        for<'a> (
            usize,
            (
                Arc<Vec<SigVal<S, V>>>,
                <ShardDataIter<'a, W, D> as Iterator>::Item,
            ),
        ): Send,
    {
        main_pl
            .item_name("shard")
            .expected_updates(Some(self.shard_edge.num_shards()))
            .display_memory(true)
            .start("Solving shards...");

        self.failed.store(false, Ordering::Relaxed);

        let result = thread_pool.scope(|_| {
            shard_iter
                .into_iter()
                .zip(data.try_chunks_mut(self.shard_edge.num_vertices()).unwrap())
                .enumerate()
                .par_bridge()
                .try_for_each_with(
                    (main_pl.clone(), pl.clone()),
                    |(main_pl, pl), (shard_index, (shard, mut data))| {
                        main_pl.info(format_args!(
                            "Analyzing shard {}/{}...",
                            shard_index + 1,
                            self.shard_edge.num_shards()
                        ));

                        // Safety: only one thread may be accessing the shard
                        let shard =
                            unsafe { &mut *(Arc::as_ptr(&shard) as *mut Vec<SigVal<S, V>>) };

                        pl.start(format!(
                            "Sorting shard {}/{}...",
                            shard_index + 1,
                            self.shard_edge.num_shards()
                        ));
                        // Sorting the signatures increases locality
                        shard.radix_sort_builder().with_low_mem_tuner().sort();
                        pl.done_with_count(shard.len());

                        if self.check_dups {
                            // Check for duplicates
                            if shard.par_windows(2).any(|w| w[0].sig == w[1].sig) {
                                return Err(SolveError::DuplicateSignature);
                            }
                        }

                        main_pl.info(format_args!(
                            "Solving shard {}/{}...",
                            shard_index + 1,
                            self.shard_edge.num_shards()
                        ));

                        if self.failed.load(Ordering::Relaxed) {
                            return Err(SolveError::UnsolvableShard);
                        }

                        if TypeId::of::<V>() == TypeId::of::<()>() {
                            // For filters, we fill the array with random data, otherwise
                            // elements with signature 0 would have a significantly higher
                            // probability of being false positives.
                            //
                            // We work around the fact that [usize] does not implement Fill
                            Mwc192::seed_from_u64(self.seed)
                                .fill_bytes(unsafe { data.as_mut_slice().align_to_mut::<u8>().1 });
                        }

                        solve_shard(self, shard_index, shard, data, pl)
                            .map_err(|_| {
                                self.failed.store(true, Ordering::Relaxed);
                                SolveError::UnsolvableShard
                            })
                            .map(|_| {
                                if !self.failed.load(Ordering::Relaxed) {
                                    main_pl.info(format_args!(
                                        "Completed shard {}/{}",
                                        shard_index + 1,
                                        self.shard_edge.num_shards()
                                    ));
                                    main_pl.update_and_display();
                                }
                            })
                    },
                )
        });

        main_pl.done();
        result
    }

    /// Peels a shard via edge indices.
    ///
    /// This peeler uses only about 10 bytes per key, but it is slower than
    /// [`VBuilder::peel_shard_by_sig_vals`], as it has to go through a
    /// cache-unfriendly memory indirection every time it has to retrieve a
    /// [`SigVal`] from the shard. It is the peeler of choice when significant
    /// parallelism is involved, or when lazy Gaussian elimination is required,
    /// as the latter requires edge indices.
    ///
    /// This method shares the same logic as [`VBuilder::peel_shard_by_sig_vals`],
    fn peel_shard_by_edge_indices<
        'a,
        V: ZeroCopy + Send + Sync,
        G: Fn(&SigVal<S, V>) -> W + Send + Sync,
    >(
        &self,
        shard_index: usize,
        shard: &'a [SigVal<S, V>],
        data: ShardData<'a, W, D>,
        get_val: &G,
        pl: &mut impl ProgressLog,
    ) -> Result<PeelResult<'a, W, D, S, V>, ()> {
        let num_vertices = self.shard_edge.num_vertices();

        pl.start(format!(
            "Generating graph for shard {}/{}...",
            shard_index + 1,
            self.shard_edge.num_shards()
        ));

        let mut xor_graph = XorGraph::<u32>::new(num_vertices);
        for (edge_index, sig_val) in shard.iter().enumerate() {
            for (side, &v) in self.shard_edge.local_edge(sig_val.sig).iter().enumerate() {
                xor_graph.add(v, edge_index as u32, side);
            }
        }
        pl.done_with_count(shard.len());

        if self.failed.load(Ordering::Relaxed) {
            return Err(());
        }

        pl.start(format!(
            "Peeling graph for shard {}/{} by edge indices...",
            shard_index + 1,
            self.shard_edge.num_shards()
        ));
        // The upper stack contains vertices to be visited. The lower stack
        // contains peeled edges. The sum of the lengths of these two items
        // cannot exceed the number of vertices.
        let mut double_stack = DoubleStack::<u32>::new(num_vertices);
        let mut sides_stack = Vec::<u8>::new();
        // Preload all vertices of degree one in the visit stack
        for (v, degree) in xor_graph.degrees().enumerate() {
            if degree == 1 {
                double_stack.push_lower(v as u32);
            }
        }

        while let Some(v) = double_stack.pop_lower() {
            let v = v as usize;
            if xor_graph.degree(v) == 0 {
                continue;
            }
            debug_assert!(xor_graph.degree(v) == 1);
            let (edge_index, side) = xor_graph.edge_index_and_side(v);
            xor_graph.zero(v);
            double_stack.push_upper(edge_index);
            sides_stack.push(side as u8);

            let e = self.shard_edge.local_edge(shard[edge_index as usize].sig);

            match side {
                0 => {
                    if xor_graph.degree(e[1]) == 2 {
                        double_stack.push_lower(e[1] as u32);
                    }
                    xor_graph.remove(e[1], edge_index, 1);
                    if xor_graph.degree(e[2]) == 2 {
                        double_stack.push_lower(e[2] as u32);
                    }
                    xor_graph.remove(e[2], edge_index, 2);
                }
                1 => {
                    if xor_graph.degree(e[0]) == 2 {
                        double_stack.push_lower(e[0] as u32);
                    }
                    xor_graph.remove(e[0], edge_index, 0);
                    if xor_graph.degree(e[2]) == 2 {
                        double_stack.push_lower(e[2] as u32);
                    }
                    xor_graph.remove(e[2], edge_index, 2);
                }
                2 => {
                    if xor_graph.degree(e[0]) == 2 {
                        double_stack.push_lower(e[0] as u32);
                    }
                    xor_graph.remove(e[0], edge_index, 0);
                    if xor_graph.degree(e[1]) == 2 {
                        double_stack.push_lower(e[1] as u32);
                    }
                    xor_graph.remove(e[1], edge_index, 1);
                }
                _ => unsafe { unreachable_unchecked() },
            }
        }

        if shard.len() != double_stack.upper_len() {
            pl.info(format_args!(
                "Peeling failed for shard {}/{} (peeled {} out of {} edges)",
                shard_index + 1,
                self.shard_edge.num_shards(),
                double_stack.upper_len(),
                shard.len(),
            ));
            return Ok(PeelResult::Partial {
                shard_index,
                shard,
                data,
                double_stack,
                sides_stack,
            });
        }
        pl.done_with_count(shard.len());

        self.assign(
            shard_index,
            data,
            double_stack
                .iter_upper()
                .map(|&edge_index| {
                    let sig_val = &shard[edge_index as usize];
                    (sig_val.sig, get_val(sig_val))
                })
                .zip(sides_stack.into_iter().rev()),
            pl,
        );

        Ok(PeelResult::Complete())
    }

    /// Peels a shard via signature/value pairs.
    ///
    /// This peeler uses about two [`SigVal`]s per key of core memory, plus a
    /// stack of bytes, but it is significantly faster than
    /// [`VBuilder::peel_shard_by_edge_indices`], as it stores directly
    /// [`SigVal`]. It is the peeler of choice when there is no significant
    /// parallelism is involved and lazy Gaussian elimination is not required,
    /// as the latter requires edge indices.
    ///
    /// This method shares the peeling code with the
    /// [`VBuilder::peel_shard_by_edge_indices`].
    fn peel_shard_by_sig_vals<
        'a,
        V: ZeroCopy + Send + Sync,
        G: Fn(&SigVal<S, V>) -> W + Send + Sync,
    >(
        &self,
        shard_index: usize,
        shard: &'a [SigVal<S, V>],
        data: ShardData<'a, W, D>,
        get_val: &G,
        pl: &mut impl ProgressLog,
    ) -> Result<(), ()>
    where
        SigVal<S, V>: BitXor + BitXorAssign + Default,
    {
        let num_vertices = self.shard_edge.num_vertices();

        pl.start(format!(
            "Generating graph for shard {}/{}...",
            shard_index + 1,
            self.shard_edge.num_shards()
        ));

        let mut xor_graph = XorGraph::<SigVal<S, V>>::new(num_vertices);
        for &sig_val in shard {
            for (side, &v) in self.shard_edge.local_edge(sig_val.sig).iter().enumerate() {
                xor_graph.add(v, sig_val, side);
            }
        }
        pl.done_with_count(shard.len());

        if self.failed.load(Ordering::Relaxed) {
            return Err(());
        }

        pl.start(format!(
            "Peeling graph for shard {}/{} by hashes...",
            shard_index + 1,
            self.shard_edge.num_shards()
        ));
        let mut sig_vals_stack = Vec::<SigVal<S, V>>::with_capacity(shard.len());
        let mut sides_stack = Vec::<u8>::with_capacity(shard.len());
        let mut visit_stack = Vec::<u32>::with_capacity(shard.len());

        // Preload all vertices of degree one in the visit stack
        for (v, degree) in xor_graph.degrees().enumerate() {
            if degree == 1 {
                visit_stack.push(v as u32);
            }
        }

        while let Some(v) = visit_stack.pop() {
            let v = v as usize;
            if xor_graph.degree(v) == 0 {
                continue;
            }
            debug_assert!(xor_graph.degree(v) == 1);
            let (sig_val, side) = xor_graph.edge_index_and_side(v);
            xor_graph.zero(v);
            sig_vals_stack.push(sig_val);
            sides_stack.push(side as u8);

            let e = self.shard_edge.local_edge(sig_val.sig);

            match side {
                0 => {
                    if xor_graph.degree(e[1]) == 2 {
                        visit_stack.push(e[1] as u32);
                    }
                    xor_graph.remove(e[1], sig_val, 1);
                    if xor_graph.degree(e[2]) == 2 {
                        visit_stack.push(e[2] as u32);
                    }
                    xor_graph.remove(e[2], sig_val, 2);
                }
                1 => {
                    if xor_graph.degree(e[0]) == 2 {
                        visit_stack.push(e[0] as u32);
                    }
                    xor_graph.remove(e[0], sig_val, 0);
                    if xor_graph.degree(e[2]) == 2 {
                        visit_stack.push(e[2] as u32);
                    }
                    xor_graph.remove(e[2], sig_val, 2);
                }
                2 => {
                    if xor_graph.degree(e[0]) == 2 {
                        visit_stack.push(e[0] as u32);
                    }
                    xor_graph.remove(e[0], sig_val, 0);
                    if xor_graph.degree(e[1]) == 2 {
                        visit_stack.push(e[1] as u32);
                    }
                    xor_graph.remove(e[1], sig_val, 1);
                }
                _ => unsafe { unreachable_unchecked() },
            }
        }

        if shard.len() != sig_vals_stack.len() {
            pl.info(format_args!(
                "Peeling failed for shard {}/{} (peeled {} out of {} edges)",
                shard_index + 1,
                self.shard_edge.num_shards(),
                sig_vals_stack.len(),
                shard.len(),
            ));
            return Err(());
        }
        pl.done_with_count(shard.len());

        self.assign(
            shard_index,
            data,
            sig_vals_stack
                .iter()
                .rev()
                .map(|sig_val| (sig_val.sig, get_val(sig_val)))
                .zip(sides_stack.iter().copied().rev()),
            pl,
        );

        Ok(())
    }

    /// Solves a shard of given index possibly using lazy Gaussian elimination,
    /// and stores the solution in the given data.
    ///
    /// As a first try, the shard is peeled. If the peeling is
    /// [partial](PeelResult::Partial), lazy Gaussian elimination is used to
    /// solve the remaining edges.
    ///
    /// This method will scan the double stack, without emptying it, to check
    /// which edges have been peeled. The information will be then passed
    /// to [`VBuilder::assign`] to complete the assignment of values.
    fn lge_shard<'a, V: ZeroCopy + Send + Sync>(
        &self,
        shard_index: usize,
        shard: &'a [SigVal<S, V>],
        data: ShardData<'a, W, D>,
        get_val: &(impl Fn(&SigVal<S, V>) -> W + Send + Sync),
        pl: &mut impl ProgressLog,
    ) -> Result<(), ()> {
        // Let's try to peel first
        match self.peel_shard_by_edge_indices(shard_index, shard, data, get_val, pl) {
            Err(()) => Err(()),
            Ok(PeelResult::Complete()) => Ok(()),
            Ok(PeelResult::Partial {
                shard_index,
                shard,
                mut data,
                double_stack,
                sides_stack,
            }) => {
                pl.info(format_args!("Switching to lazy Gaussian elimination..."));
                // Likely result--we have solve the rest
                pl.start(format!(
                    "Generating system for shard {}/{}...",
                    shard_index + 1,
                    self.shard_edge.num_shards()
                ));

                let num_vertices = self.shard_edge.num_vertices();
                let mut peeled_edges = BitVec::new(shard.len());
                let mut used_vars = BitVec::new(num_vertices);
                for &edge in double_stack.iter_upper() {
                    peeled_edges.set(edge as _, true);
                }

                // Create data for an F₂ system using non-peeled edges
                //
                // SAFETY: there is no undefined behavior here, but the
                // raw construction methods we use assume that the
                // equations are sorted, that the variables are not repeated,
                // and the variables are in the range [0..num_vertices).
                let mut system = unsafe {
                    crate::utils::mod2_sys::Modulo2System::from_parts(
                        num_vertices,
                        shard
                            .iter()
                            .enumerate()
                            .filter(|(edge_index, _)| !peeled_edges[*edge_index])
                            .map(|(_edge_index, sig_val)| {
                                let mut eq: Vec<_> = self
                                    .shard_edge
                                    .local_edge(sig_val.sig)
                                    .iter()
                                    .map(|&x| {
                                        used_vars.set(x, true);
                                        x as u32
                                    })
                                    .collect();
                                eq.sort_unstable();
                                crate::utils::mod2_sys::Modulo2Equation::from_parts(
                                    eq,
                                    get_val(sig_val),
                                )
                            })
                            .collect(),
                    )
                };

                if self.failed.load(Ordering::Relaxed) {
                    return Err(());
                }

                pl.expected_updates(Some(system.num_equations()));
                pl.start("Solving system...");
                let result = system.lazy_gaussian_elimination().map_err(|_| ())?;
                pl.done_with_count(system.num_equations());

                for (v, &value) in result.iter().enumerate().filter(|(v, _)| used_vars[*v]) {
                    data.set(v, value);
                }

                self.assign(
                    shard_index,
                    data,
                    double_stack
                        .iter_upper()
                        .map(|&edge_index| {
                            let sig_val = &shard[edge_index as usize];
                            (sig_val.sig, get_val(sig_val))
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
    /// `sig_vals_sides` is an iterator returning pairs of signature/value pairs
    /// and sides in reverse peeling order.
    fn assign(
        &self,
        shard_index: usize,
        mut data: ShardData<'_, W, D>,
        sigs_vals_sides: impl Iterator<Item = ((S, W), u8)>,
        pl: &mut impl ProgressLog,
    ) {
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
            unsafe {
                let xor = match side {
                    0 => data.get_unchecked(edge[1]) ^ data.get_unchecked(edge[2]),
                    1 => data.get_unchecked(edge[0]) ^ data.get_unchecked(edge[2]),
                    2 => data.get_unchecked(edge[0]) ^ data.get_unchecked(edge[1]),
                    _ => core::hint::unreachable_unchecked(),
                };

                data.set_unchecked(edge[side], val ^ xor);
            }
            pl.light_update();
        }
        pl.done();
    }
}
