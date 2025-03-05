/*
*
* SPDX-FileCopyrightText: 2023 Sebastiano Vigna
*
* SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
*/

use super::vfunc::*;
use crate::bits::*;
use crate::prelude::Rank9;
use crate::traits::bit_field_slice::{BitFieldSlice, BitFieldSliceMut, Word};
use crate::traits::NumBits;
use crate::utils::*;
use common_traits::CastableInto;
use derivative::Derivative;
use derive_setters::*;
use dsi_progress_logger::*;
use epserde::prelude::*;
use pluralizer::pluralize;
use rand::rngs::SmallRng;
use rand::Rng;
use rand::SeedableRng;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use rdst::*;
use std::any::TypeId;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

const LOG2_MAX_SHARDS: u32 = 12;

/// A set of edge indices and sides represented by a 64-bit integer.
///
/// Each (*k*-hyper)edge is a set of *k* vertices (by construction fuse graphs
/// to not have degenerate edges), but we represent it internally as a vector.
/// We call *side* the position of a vertex in the edge.
///
/// For each vertex, we store both the indices of the edges incident to the
/// vertex and the sides of the vertex in such edges. While technically not
/// necessary to perform peeling, the knowledge of the sides speeds up the
/// peeling breadth-first visit by reducing the number of tests that are
/// necessary to update the degrees once the edge is peeled (see the
/// `peel_shard` method).
///
/// Edge indices and sides are packed together using Djamal's XOR trick (see
/// “Cache-Oblivious Peeling of Random Hypergraphs”,
/// https://doi.org/10.1109/DCC.2014.48): since during the peeling breadth-first
/// visit we need to know the content of the list only when a single edge index
/// is present, we can XOR together all the edge indices and sides.
///
/// The lower SIDE_SHIFT bits contain a XOR of the edge indices; the next two
/// bits contain a XOR of the sides; the remaining upper bits contain the
/// degree. The degree can be stored with a small number of bits because the
/// graph is random, so the maximum degree is *O*(log log *n*).
///
/// When we peel an edge, we do not remove its index from the list of the vertex
/// it was peeled from, but rather we zero the degree. This allows us to store
/// the peeled edge and its side in the list.
///
/// This approach reduces the core memory usage for the hypergraph to a `u64`
/// per vertex. Edges are derived on the fly from signatures using the edge
/// indices. The breadth-first queue and the stack of peeled edges can be
/// represented just using vertices, as the edge indices can be
/// retrieved from this list.
#[derive(Debug, Default)]
struct EdgeIndexSideSet(u64);
impl EdgeIndexSideSet {
    const DEG_SHIFT: u32 = u64::BITS - 16;
    const SIDE_SHIFT: u32 = u64::BITS - 18;
    const SIDE_MASK: usize = 3;
    const EDGE_INDEX_MASK: u64 = (1_u64 << Self::SIDE_SHIFT) - 1;
    const DEG: u64 = 1_u64 << Self::DEG_SHIFT;
    const MAX_DEG: usize = (u64::MAX >> Self::DEG_SHIFT) as usize;

    /// Add an edge index and a side to the set.
    #[inline(always)]
    fn add(&mut self, edge_index: usize, side: usize) {
        debug_assert!(self.degree() < Self::MAX_DEG);
        self.0 += Self::DEG;
        self.0 ^= (side as u64) << Self::SIDE_SHIFT | edge_index as u64;
    }

    /// Remove an edge index and a side from the set.
    #[inline(always)]
    fn remove(&mut self, edge_index: usize, side: usize) {
        debug_assert!(self.degree() > 0);
        self.0 -= Self::DEG;
        self.0 ^= (side as u64) << Self::SIDE_SHIFT | edge_index as u64;
    }

    /// Return the degree of the vertex.
    ///
    /// When the degree is zero, the list stores a peeled edge and the
    /// associated side.
    #[inline(always)]
    fn degree(&self) -> usize {
        (self.0 >> Self::DEG_SHIFT) as usize
    }

    /// Retrieve an edge index and its side.
    ///
    /// This method return meaningful values only when the degree is zero (i.e.,
    /// the edge has been peeled and we use the list to store it) or one (i.e.,
    /// the edge is peelable and we can retrieved as it is the only vertex).
    #[inline(always)]
    fn edge_index_and_side(&self) -> (usize, usize) {
        debug_assert!(self.degree() < 2);
        (
            (self.0 & EdgeIndexSideSet::EDGE_INDEX_MASK) as usize,
            (self.0 >> Self::SIDE_SHIFT) as usize & Self::SIDE_MASK,
        )
    }

    /// Zero the degree of the vertex.
    ///
    /// This method should be called when the degree is one, and the list is
    /// used to store a peeled edge and the associated side.
    #[inline(always)]
    fn zero(&mut self) {
        debug_assert!(self.degree() == 1);
        self.0 &= (1 << Self::DEG_SHIFT) - 1;
    }
}

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
/// shard of the signature/value pairs; for each signature/value in a shard, a
/// thread will allocate about two `usize` values in core memory; in the case of
/// offline construction, also signatures and values in the shard will be stored
/// in core memory.
///
/// For very large key sets shards will be significantly smaller than the number
/// of keys, so the memory usage, in particular in offline mode, can be
/// significantly reduced. Note that using too many threads might actually be
/// harmful due to memory contention.
///
/// The generic parameters are explained in the [`VFunc`] documentation. You
/// have to choose the type of the output values and the backend. The remaining
/// parameters have default values that are the same as those of
/// [`VFunc`]/[`VFilter`].
///
/// All construction methods require to pass one or two [`RewindableIoLender`],
/// and the construction might fail and keys might be scanned again. The
/// structures in the [`lenders`] modules provide easy ways to build such
/// lenders, even starting from compressed files of UTF-8 strings. The type of
/// the keys of the resulting filter or function will be the type of the
/// elements of the [`RewindableIoLender`].
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
    E: ShardEdge<S, 3> = Fuse3Shards,
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

    /// Use radix sort to check for duplicated signatures.
    #[setters(generate = true)]
    check_dups: bool,

    /// The seed for the random number generator.
    #[setters(generate = true)]
    seed: u64,

    /// The base-2 logarithm of buckets of the [`SigStore`]. The default is 8.
    /// This value is automatically set if you provide an expected number of
    /// keys, which makes the construction faster.
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
    /// Fast-stop for failed attemps.
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

type ShardIter<'a, W, D> = <D as BitFieldSliceMut<W>>::ChunksMut<'a>;
type Shard<'a, W, D> = <ShardIter<'a, W, D> as Iterator>::Item;

impl<
        W: ZeroCopy + Word + Send + Sync,
        D: BitFieldSlice<W> + BitFieldSliceMut<W> + Send + Sync,
        S: Sig + ZeroCopy + Send + Sync,
        E: ShardEdge<S, 3>,
    > VBuilder<W, D, S, E>
{
    /// Solve in parallel the shards returned by the given iterator.
    ///
    /// # Arguments
    ///
    /// * `shard_iter`: an iterator returning the shards to solve.
    ///
    /// * `data`: the storage for the solution values.
    ///
    /// * `new_data`: a function to create shard-local storage for the values.
    ///
    /// * `solve_shard`: a method to solve a shard; it takes the shard index,
    ///   the shard, shard-local storage, and a progress logger, and returns the
    ///   shard index and the shard-local storage filled with a solution, or an
    ///   error.
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
        solve_shard: impl Fn(
                &Self,
                usize,
                &'b mut Vec<SigVal<S, V>>,
                Shard<'b, W, D>,
                &mut P,
            ) -> Result<usize, ()>
            + Send
            + Sync,
        thread_pool: &rayon::ThreadPool,
        main_pl: &mut C,
        pl: &mut P,
    ) -> Result<(), SolveError>
    where
        I::IntoIter: Send,
        SigVal<S, V>: RadixKey + Send + Sync,
        for<'a> ShardIter<'a, W, D>: Send,
        for<'a> std::iter::Enumerate<std::iter::Zip<<I as IntoIterator>::IntoIter, ShardIter<'a, W, D>>>:
            Send,
        for<'a> (
            usize,
            (
                Arc<Vec<SigVal<S, V>>>,
                <ShardIter<'a, W, D> as Iterator>::Item,
            ),
        ): Send,
    {
        // TODO: optimize for the non-parallel case
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
                    |(main_pl, pl), (shard_index, (shard, data))| {
                        main_pl.info(format_args!(
                            "Analyzing shard {}/{}...",
                            shard_index + 1,
                            self.shard_edge.num_shards()
                        ));

                        // Safety: only one thread may be accessing the shard
                        let shard =
                            unsafe { &mut *(Arc::as_ptr(&shard) as *mut Vec<SigVal<S, V>>) };

                        if TypeId::of::<V>() == TypeId::of::<()>()
                            && TypeId::of::<S>() == TypeId::of::<[u64; 1]>()
                            && self.num_keys >= 100_000_000
                        {
                            // Filters using 64-bit hashes need special
                            // treatment because duplicates can happen and can
                            // be ignored. Below the size limit the probability
                            // of a duplicate is less than 0.000391.
                            shard.radix_sort_builder().with_low_mem_tuner().sort();
                            // WARNING: we are screwing the internal state of
                            // the SigSorter (the chunk sizes won't be correct
                            // anymore), but we don't care because we are going
                            // to throw it away.
                            shard.dedup_by_key(|x| x.sig);
                        } else if self.check_dups {
                            // Check for duplicates
                            shard.radix_sort_builder().with_low_mem_tuner().sort();

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

                        solve_shard(self, shard_index, shard, data, pl)
                            .map_err(|_| {
                                self.failed.store(true, Ordering::Relaxed);
                                SolveError::UnsolvableShard
                            })
                            .map(|shard_index| {
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

    /// Solve a shard by peeling.
    ///
    /// Return the shard index and the shard-local data, in case of success, or
    /// the shard index, the shard, the edge lists, the shard-local data, and
    /// the stack at the end of the peeling procedure in case of failure. These
    /// data can be then passed to a linear solver to complete the solution.
    fn peel_shard<'a, V: ZeroCopy + Send + Sync, G: Fn(&SigVal<S, V>) -> W + Send + Sync>(
        &self,
        shard_index: usize,
        shard: &'a Vec<SigVal<S, V>>,
        data: Shard<'a, W, D>,
        get_val: &G,
        pl: &mut impl ProgressLog,
    ) -> Result<
        usize,
        (
            usize,
            &'a Vec<SigVal<S, V>>,
            Shard<'a, W, D>,
            Vec<EdgeIndexSideSet>,
            Vec<usize>,
        ),
    > {
        if self.failed.load(Ordering::Relaxed) {
            return Err((shard_index, shard, data, vec![], vec![]));
        }

        let num_vertices = self.shard_edge.num_vertices();

        pl.start(format!(
            "Generating graph for shard {}/{}...",
            shard_index + 1,
            self.shard_edge.num_shards()
        ));

        let mut edge_sets = Vec::new();
        edge_sets.resize_with(num_vertices, EdgeIndexSideSet::default);
        for (edge_index, sig_val) in shard.iter().enumerate() {
            for (side, &v) in self.shard_edge.local_edge(sig_val.sig).iter().enumerate() {
                edge_sets[v].add(edge_index, side);
            }
        }
        pl.done_with_count(shard.len());

        if self.failed.load(Ordering::Relaxed) {
            return Err((shard_index, shard, data, edge_sets, vec![]));
        }

        pl.start(format!(
            "Peeling graph for shard {}/{}...",
            shard_index + 1,
            self.shard_edge.num_shards()
        ));
        let mut stack = vec![0; num_vertices + 1];
        // Preload all vertices of degree one in the visit queue
        let mut top = 0;
        for v in 0..num_vertices {
            stack[top] = v;
            if edge_sets[v].degree() == 1 {
                top += 1;
            }
        }
        let (mut pos, mut curr) = (0, 0);
        // This array, indexed by the side, gives the other two sides
        let other_side = [1, 2, 0, 1];
        // This array will be loaded with the vertices corresponding to
        // the sides in other_side
        let mut other_vertex = [0; 4];

        while pos < top {
            let v = stack[pos];
            pos += 1;
            if edge_sets[v].degree() == 0 {
                continue;
            }
            let (edge_index, side) = edge_sets[v].edge_index_and_side();
            edge_sets[v].zero();
            stack[curr] = v;
            curr += 1;

            let e = self.shard_edge.local_edge(shard[edge_index].sig);

            (other_vertex[0], other_vertex[1]) = (e[1], e[2]);
            (other_vertex[2], other_vertex[3]) = (e[0], e[1]);

            edge_sets[other_vertex[side]].remove(edge_index, other_side[side]);
            stack[top] = other_vertex[side];
            if edge_sets[other_vertex[side]].degree() == 1 {
                top += 1;
            }

            edge_sets[other_vertex[side + 1]].remove(edge_index, other_side[side + 1]);
            stack[top] = other_vertex[side + 1];
            if edge_sets[other_vertex[side + 1]].degree() == 1 {
                top += 1;
            }
        }
        debug_assert!(top <= num_vertices);
        stack.truncate(curr);
        if shard.len() != stack.len() {
            pl.info(format_args!(
                "Peeling failed for shard {}/{} (peeled {} out of {} edges)",
                shard_index + 1,
                self.shard_edge.num_shards(),
                stack.len(),
                shard.len(),
            ));
            return Err((shard_index, shard, data, edge_sets, stack));
        }
        pl.done_with_count(shard.len());

        Ok(self.assign(shard_index, shard, data, get_val, edge_sets, stack, pl))
    }

    /// Perform assignment of values based on peeling data.
    ///
    /// This method might be called after a successful peeling procedure, or
    /// after a linear solver has been used to solve the remaining edges.
    fn assign<'a, V: ZeroCopy + Send + Sync>(
        &self,
        shard_index: usize,
        shard: &Vec<SigVal<S, V>>,
        mut data: Shard<'a, W, D>,
        get_val: &(impl Fn(&SigVal<S, V>) -> W + Send + Sync),
        edge_sets: Vec<EdgeIndexSideSet>,
        mut stack: Vec<usize>,
        pl: &mut impl ProgressLog,
    ) -> usize {
        if self.failed.load(Ordering::Relaxed) {
            return shard_index;
        }

        pl.start(format!(
            "Assigning values for shard {}/{}...",
            shard_index + 1,
            self.shard_edge.num_shards()
        ));
        while let Some(v) = stack.pop() {
            // Assignments after linear solving must skip unpeeled edges
            if edge_sets[v].degree() != 0 {
                continue;
            }
            let (edge_index, side) = edge_sets[v].edge_index_and_side();
            let edge = self.shard_edge.local_edge(shard[edge_index].sig);
            unsafe {
                let value = match side {
                    0 => data.get_unchecked(edge[1]) ^ data.get_unchecked(edge[2]),
                    1 => data.get_unchecked(edge[0]) ^ data.get_unchecked(edge[2]),
                    2 => data.get_unchecked(edge[0]) ^ data.get_unchecked(edge[1]),
                    _ => unreachable!(),
                };

                data.set_unchecked(v, get_val(&shard[edge_index]) ^ value);
            }
            debug_assert_eq!(
                data.get(edge[0]) ^ data.get(edge[1]) ^ data.get(edge[2]),
                get_val(&shard[edge_index])
            );
        }
        pl.done_with_count(shard.len());

        shard_index
    }

    /// Solve a shard of given index using lazy Gaussian elimination, and store
    /// the solution in the given data.
    ///
    /// Return the shard index and the data, in case of success, or `Err(())` in
    /// case of failure.
    fn lge_shard<'a, V: ZeroCopy + Send + Sync>(
        &self,
        shard_index: usize,
        shard: &'a Vec<SigVal<S, V>>,
        data: Shard<'a, W, D>,
        get_val: &(impl Fn(&SigVal<S, V>) -> W + Send + Sync),
        pl: &mut impl ProgressLog,
    ) -> Result<usize, ()> {
        // Let's try to peel first
        match self.peel_shard(shard_index, shard, data, get_val, pl) {
            Ok(shard_index) => {
                // Unlikely result, but we're happy if it happens
                Ok(shard_index)
            }
            Err((shard_index, shard, mut data, edge_sets, stack)) => {
                pl.info(format_args!("Switching to lazy Gaussian elimination..."));
                // Likely result--we have solve the rest
                pl.start(format!(
                    "Generating system for shard {}/{}...",
                    shard_index + 1,
                    self.shard_edge.num_shards()
                ));

                // Build a ranked vector of unpeeled equation
                let mut unpeeled = bit_vec![true; shard.len()];
                stack
                    .iter()
                    .filter(|&v| edge_sets[*v].degree() == 0)
                    .for_each(|&v| {
                        unpeeled.set(edge_sets[v].edge_index_and_side().0, false);
                    });
                let unpeeled = Rank9::new(unpeeled);

                // Create data for an F₂ system using non-peeled edges
                let mut system = unsafe {
                    crate::utils::mod2_sys_sparse::Modulo2System::from_parts(
                        self.shard_edge.num_vertices(),
                        shard
                            .iter()
                            .enumerate()
                            .filter(|(edge_index, _)| unpeeled[*edge_index])
                            .map(|(_edge_index, sig_val)| {
                                let mut eq: Vec<_> = self
                                    .shard_edge
                                    .local_edge(sig_val.sig)
                                    .iter()
                                    .map(|x| *x as u32)
                                    .collect();
                                eq.sort_unstable();
                                crate::utils::mod2_sys_sparse::Modulo2Equation::from_parts(
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

                pl.expected_updates(Some(unpeeled.num_ones()));
                pl.start("Solving system...");
                let result = system.lazy_gaussian_elimination().map_err(|_| ())?;
                pl.done_with_count(unpeeled.num_ones());

                for (v, &value) in result.iter().enumerate() {
                    data.set(v, value);
                }

                Ok(self.assign(shard_index, shard, data, get_val, edge_sets, stack, pl))
            }
        }
    }
}

/// Builds a new function using a `Box<[W]>` to store values.
///
/// Since values are stored in a slice, access is particularly fast, but the bit
/// width of the output of the function will be  exactly the bit width of `W`.
impl<W: ZeroCopy + Word, S: Sig + Send + Sync, E: ShardEdge<S, 3>> VBuilder<W, Box<[W]>, S, E>
where
    SigVal<S, W>: RadixKey + Send + Sync,
    Box<[W]>: BitFieldSliceMut<W> + BitFieldSlice<W>,
{
    pub fn try_build_func<T: ?Sized + ToSig<S>>(
        mut self,
        keys: impl RewindableIoLender<T>,
        values: impl RewindableIoLender<W>,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> anyhow::Result<VFunc<T, W, Box<[W]>, S, E>>
    where
        for<'a> ShardIter<'a, W, Box<[W]>>: Send,
        for<'a> Shard<'a, W, Box<[W]>>: Send,
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
    SigVal<S, ()>: RadixKey + Send + Sync,
    Box<[W]>: BitFieldSliceMut<W> + BitFieldSlice<W>,
    u64: CastableInto<W>,
{
    pub fn try_build_filter<T: ?Sized + ToSig<S>>(
        mut self,
        keys: impl RewindableIoLender<T>,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> anyhow::Result<VFilter<W, VFunc<T, W, Box<[W]>, S, E>>>
    where
        for<'a> ShardIter<'a, W, Box<[W]>>: Send,
        for<'a> Shard<'a, W, Box<[W]>>: Send,
    {
        let filter_mask = W::MAX;
        let get_val = |sig_val: &SigVal<S, ()>| sig_val.sig.sig_u64().cast();
        let new_data = |_bit_width: usize, len: usize| vec![W::ZERO; len].into();

        Ok(VFilter {
            func: self.build_loop(
                keys,
                FromIntoIterator::from(itertools::repeat_n((), usize::MAX)),
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
    SigVal<S, W>: RadixKey + Send + Sync,
{
    pub fn try_build_func<T: ?Sized + ToSig<S>>(
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
/// They must be in any case at most the bit width of `W`.
///
/// Typically `W` will be `usize` or `u64`.
impl<W: ZeroCopy + Word, S: Sig + Send + Sync, E: ShardEdge<S, 3>> VBuilder<W, BitFieldVec<W>, S, E>
where
    SigVal<S, ()>: RadixKey + Send + Sync,
    Box<[W]>: BitFieldSliceMut<W> + BitFieldSlice<W>,
    u64: CastableInto<W>,
{
    pub fn try_build_filter<T: ?Sized + ToSig<S>>(
        mut self,
        keys: impl RewindableIoLender<T>,
        filter_bits: u32,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> anyhow::Result<VFilter<W, VFunc<T, W, BitFieldVec<W>, S, E>>> {
        assert!(filter_bits > 0);
        assert!(filter_bits <= W::BITS as u32);
        let filter_mask = W::MAX >> (W::BITS as u32 - filter_bits);
        let get_val = |sig_val: &SigVal<S, ()>| sig_val.sig.sig_u64().cast() & filter_mask;
        let new_data = |bit_width, len| BitFieldVec::<W>::new(bit_width, len);

        Ok(VFilter {
            func: self.build_loop(
                keys,
                FromIntoIterator::from(itertools::repeat_n((), usize::MAX)),
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
    /// Build and return a new function with given keys and values.
    ///
    /// This function can build functions based both on vectors and on bit-field
    /// vectors. The necessary abstraction is provided by the `new(bit_width,
    /// len)` function, which is called to create the data structure to store
    /// the values.
    fn build_loop<T: ?Sized + ToSig<S>, V: ZeroCopy + Send + Sync>(
        &mut self,
        mut keys: impl RewindableIoLender<T>,
        mut values: impl RewindableIoLender<V>,
        get_val: &(impl Fn(&SigVal<S, V>) -> W + Send + Sync),
        new: fn(usize, usize) -> D,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> anyhow::Result<VFunc<T, W, D, S, E>>
    where
        SigVal<S, V>: RadixKey + Send + Sync,
        for<'a> ShardIter<'a, W, D>: Send,
        for<'a> <ShardIter<'a, W, D> as Iterator>::Item: Send,
    {
        let mut dup_count = 0;
        let start = Instant::now();
        let mut prng = SmallRng::seed_from_u64(self.seed);

        // Loop until success or duplicate detection
        loop {
            let seed = prng.random();
            pl.item_name("key");
            pl.start("Reading input...");

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
                    new,
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
                    new,
                    pl,
                )
            } {
                Ok(func) => {
                    pl.info(format_args!(
                        "Completed in {:.3} seconds ({:.3} ns/key)",
                        start.elapsed().as_secs_f64(),
                        start.elapsed().as_nanos() as f64 / self.num_keys as f64
                    ));
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
    fn try_seed<
        T: ?Sized + ToSig<S>,
        V: ZeroCopy + Send + Sync,
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
        SigVal<S, V>: RadixKey + Send + Sync,
        for<'a> ShardIter<'a, W, D>: Send,
        for<'a> <ShardIter<'a, W, D> as Iterator>::Item: Send,
    {
        let mut max_value = W::ZERO;

        if let Some(expected_num_keys) = self.expected_num_keys {
            self.shard_edge.set_up_shards(expected_num_keys);
            self.log2_buckets = self.shard_edge.shard_high_bits();
        }

        let num_buckets = 1 << self.log2_buckets;
        pl.info(format_args!(
            "Using {}",
            pluralize("bucket", num_buckets, true)
        ));

        while let Some(result) = keys.next() {
            match result {
                Ok(key) => {
                    pl.light_update();
                    // This might be an actual value, if we are building a
                    // function, or (), if we are building a filter.
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
            Err(SolveError::MaxShardTooBig.into())
        } else {
            #[cfg(not(feature = "vbuilder_no_data"))]
            let data = new_data(
                self.bit_width,
                self.shard_edge.num_vertices() * self.shard_edge.num_shards(),
            );
            #[cfg(feature = "vbuilder_no_data")]
            let data = new_data(self.bit_width, 0);
            self.try_build_from_shard_iter(seed, data, shard_store.iter(), get_val, pl)
                .map_err(Into::into)
        }
    }

    /// Build and return a new function starting from an iterator on shards.
    ///
    /// This method provide construction logic that is independent from the
    /// actual storage of the values (offline or in core memory.)
    ///
    /// See [`VBuilder::_build`] for more details on the parameters.
    fn try_build_from_shard_iter<
        T: ?Sized + ToSig<S>,
        I,
        P,
        V: ZeroCopy + Send + Sync,
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
        SigVal<S, V>: RadixKey,
        P: ProgressLog + Clone + Send + Sync,
        I: Iterator<Item = Arc<Vec<SigVal<S, V>>>> + Send,
        for<'a> ShardIter<'a, W, D>: Send,
        for<'a> std::iter::Enumerate<std::iter::Zip<<I as IntoIterator>::IntoIter, ShardIter<'a, W, D>>>:
            Send,
        for<'a> (
            usize,
            (
                Arc<Vec<SigVal<S, V>>>,
                <ShardIter<'a, W, D> as Iterator>::Item,
            ),
        ): Send,
    {
        let thread_pool = ThreadPoolBuilder::new()
            .num_threads(self.shard_edge.num_shards().min(self.max_num_threads))
            .build()
            .unwrap(); // Seroiusly, it's not going to fail

        pl.info(format_args!("{}", self.shard_edge));
        pl.info(format_args!(
            "c: {}, Overhead: {:.4}% Number of threads: {}",
            self.c,
            (self.shard_edge.num_vertices() * self.shard_edge.num_shards()) as f64
                / (self.num_keys as f64),
            thread_pool.current_num_threads()
        ));

        if self.lge {
            pl.info(format_args!("Switching to lazy Gaussian elimination"));
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
        } else {
            self.par_solve(
                shard_iter,
                &mut data,
                |this, shard_index, shard, data, pl| {
                    this.peel_shard(shard_index, shard, data, get_val, pl)
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

#[cfg(test)]
mod tests {
    use crate::func::vbuilder::EdgeIndexSideSet;
    use rayon::iter::{IntoParallelIterator, ParallelIterator};
    use xxhash_rust::xxh3;

    fn _test_peeling_c(n: usize, c: f64, log2_seg_size: u32, seed: u64) -> usize {
        let l = ((c * n as f64).ceil() as usize)
            .div_ceil(1 << log2_seg_size)
            .saturating_sub(2)
            .max(1)
            .try_into()
            .unwrap();
        _test_peeling_l(n, l, log2_seg_size, seed)
    }

    fn _test_peeling_l(n: usize, l: usize, log2_seg_size: u32, seed: u64) -> usize {
        fn edge(l: usize, log2_seg_size: u32, sig: &[u64; 2]) -> [usize; 3] {
            let first_segment = (((sig[0] >> 32) * l as u64) >> 32) as usize;
            let start = first_segment << log2_seg_size;
            let segment_size = 1 << log2_seg_size;
            let segment_mask = segment_size - 1;

            [
                (sig[0] as usize & segment_mask) + start,
                ((sig[1] >> 32) as usize & segment_mask) + start + segment_size,
                (sig[1] as usize & segment_mask) + start + 2 * segment_size,
            ]
        }

        fn sig(i: usize, seed: u64) -> [u64; 2] {
            let hash128 = xxh3::xxh3_128_with_seed((i as u64).to_ne_bytes().as_ref(), seed);
            [(hash128 >> 64) as u64, hash128 as u64]
        }

        let num_vertices = (l + 2) << log2_seg_size;

        let mut edge_sets = Vec::new();
        edge_sets.resize_with(num_vertices, EdgeIndexSideSet::default);

        for i in 0..n {
            for (side, &v) in edge(l, log2_seg_size, &sig(i, seed)).iter().enumerate() {
                edge_sets[v].add(i, side);
            }
        }

        let mut stack = Vec::new();
        // Breadth-first visit in reverse order TODO: check if this is the best order
        for v in (0..num_vertices).rev() {
            if edge_sets[v].degree() != 1 {
                continue;
            }
            let mut pos = stack.len();
            let mut curr = stack.len();
            stack.push(v);
            while pos < stack.len() {
                let v = stack[pos];
                pos += 1;
                if edge_sets[v].degree() == 0 {
                    continue; // Skip no longer useful entries
                }
                let (edge_index, side) = edge_sets[v].edge_index_and_side();
                edge_sets[v].zero();
                stack[curr] = v;
                curr += 1;

                let e = edge(l, log2_seg_size, &sig(edge_index, seed));
                // Remove edge from the lists of the other two vertices
                match side {
                    0 => {
                        edge_sets[e[1]].remove(edge_index, 1);
                        if edge_sets[e[1]].degree() == 1 {
                            stack.push(e[1]);
                        }
                        edge_sets[e[2]].remove(edge_index, 2);
                        if edge_sets[e[2]].degree() == 1 {
                            stack.push(e[2]);
                        }
                    }
                    1 => {
                        edge_sets[e[0]].remove(edge_index, 0);
                        if edge_sets[e[0]].degree() == 1 {
                            stack.push(e[0]);
                        }
                        edge_sets[e[2]].remove(edge_index, 2);
                        if edge_sets[e[2]].degree() == 1 {
                            stack.push(e[2]);
                        }
                    }
                    2 => {
                        edge_sets[e[0]].remove(edge_index, 0);
                        if edge_sets[e[0]].degree() == 1 {
                            stack.push(e[0]);
                        }
                        edge_sets[e[1]].remove(edge_index, 1);
                        if edge_sets[e[1]].degree() == 1 {
                            stack.push(e[1]);
                        }
                    }
                    _ => unreachable!("{}", side),
                }
            }
            stack.truncate(curr);
        }
        stack.len()
    }

    //#[test]
    fn test_peeling() {
        fn fuse_log2_seg_size(arity: usize, n: usize) -> u32 {
            match arity {
                3 => ((n.max(1) as f64).ln() / (3.33_f64).ln() + 2.25).floor() as u32,
                _ => unimplemented!(),
            }
        }

        let mut size = 6938893866 * 5 / 4;

        loop {
            let std_log2_seg_size = fuse_log2_seg_size(3, size);

            'outer: for log2_seg_size in std_log2_seg_size + 6..std_log2_seg_size + 9 {
                for seed in 0..5 {
                    eprintln!("Testing {size} {log2_seg_size} (estimate: {std_log2_seg_size})...");
                    let peeled = _test_peeling_c(size, 1.105, log2_seg_size, seed);
                    let success = size == peeled;
                    println!("{size} {log2_seg_size} {peeled} {success}");
                    if !success {
                        break 'outer;
                    }
                }
            }

            size = size * 5 / 4;
        }
    }
    //#[test]
    fn explore_peeling() {
        fn fuse_log2_seg_size(arity: usize, n: usize) -> u32 {
            match arity {
                3 => ((n.max(1) as f64).ln() / (3.33_f64).ln() + 2.25).floor() as u32,
                _ => unimplemented!(),
            }
        }

        let mut size = 1 << 20;

        loop {
            for c in (1105..=1150).step_by(5) {
                let c = c as f64 / 1000.0;
                let base_log2_seg_size = fuse_log2_seg_size(3, size);
                for log2_seg_size in base_log2_seg_size.saturating_sub(3)..base_log2_seg_size + 3 {
                    let failures: usize = (0..100)
                        .into_par_iter()
                        .map(|seed| (!_test_peeling_c(size, c, log2_seg_size, seed)) as usize)
                        .sum();
                    eprintln!("{size} {c} {log2_seg_size} {failures}");
                }
            }
            size = size * 5 / 4;
        }
    }

    //#[test]
    fn test_peelability() {
        fn fuse_log2_seg_size(arity: usize, n: usize) -> u32 {
            match arity {
                3 => ((n.max(1) as f64).ln() / (3.33_f64).ln() + 2.25).floor() as u32,
                _ => unimplemented!(),
            }
        }

        let mut size = 40_usize;

        loop {
            let mut peels = vec![];
            for log2_seg_size in 0..20 {
                let l = (1..1000)
                    .filter(|l| ((l + 2) << log2_seg_size) as f64 / size as f64 <= 1.12)
                    .max();

                if let Some(l) = l {
                    let mut tot_peeled = 0;
                    for seed in 0..10 {
                        let peeled = _test_peeling_l(size, l, log2_seg_size as u32, seed);
                        tot_peeled += peeled;
                    }

                    eprintln!(
                        "{size}\t{}\t{}\t{}%",
                        log2_seg_size,
                        l,
                        100.0 * tot_peeled as f64 / (size as f64 * 10.)
                    );
                    peels.push((l, log2_seg_size, tot_peeled));
                }
            }

            peels.sort_by(|(_, _, tot_peeled1), (_, _, tot_peeled2)| tot_peeled2.cmp(tot_peeled1));

            peels[..3]
                .iter()
                .for_each(|&(l, log2_seg_size, tot_peeled)| {
                    println!(
                        "{size}\t{}\t{}\t{}%",
                        l,
                        log2_seg_size,
                        100.0 * tot_peeled as f64 / (size as f64 * 10.)
                    );
                });

            size = size * 20 / 19;
        }
    }
}
