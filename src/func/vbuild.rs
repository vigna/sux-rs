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
use crate::traits::Rank;
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
/// The output type `W` can be selected to be any of the unsigned integer types;
/// the default is `usize`.
///
/// There are two construction modes: in core memory (default) and
/// [offline](VBuilder::offline). In the first case, space will be allocated
/// in core memory for signatures and associated values for all keys; in
/// the second case, such information will be stored in a number of on-disk
/// buckets using a [`SigStore`].
///
/// Once signatures have been computed, each parallel thread will process a
/// shard of the signatures; for each signature in a shard, a thread will
/// allocate about two `usize` values in core memory; in the case of offline
/// construction, also signatures and values in the shard will be stored in core
/// memory.
///
/// For very large key sets shards will be significantly smaller than the number
/// of keys, so the memory usage, in particular in offline mode, can be
/// significantly reduced. Note that using too many threads might actually be
/// harmful due to memory contention: eight is usually a good value.
#[derive(Setters, Debug, Derivative)]
#[derivative(Default)]
#[setters(generate = false)]
pub struct VBuilder<
    T: ?Sized + Send + Sync + ToSig<S>,
    W: ZeroCopy + Word,
    D: BitFieldSlice<W> + Send + Sync = BitFieldVec<W>,
    S = [u64; 2],
    E: ShardEdge<S, 3> = FuseShards,
    V = W,
> {
    /// The optional expected number of keys.
    #[setters(generate = true, strip_option)]
    #[derivative(Default(value = "None"))]
    expected_num_keys: Option<usize>,

    /// The maximum number of parallel threads to use.
    #[setters(generate = true)]
    #[derivative(Default(value = "8"))]
    max_num_threads: usize,

    /// Use disk-based buckets to reduce core memory usage at construction time.
    #[setters(generate = true)]
    offline: bool,

    /// The seed for the random number generator.
    #[setters(generate = true)]
    seed: u64,

    /// The base-2 logarithm of the number of on-disk buckets. Used only if
    /// `offline` is `true`. The default is 8.
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
    /// Whether we are using lazy Gaussian elimination.
    lazy_gaussian: bool,
    /// Fast-stop for failed attemps.
    failed: AtomicBool,
    #[doc(hidden)]
    _marker_t: PhantomData<T>,
    #[doc(hidden)]
    _marker_v: PhantomData<(V, W, D, S)>,
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

impl<
        T: ?Sized + Send + Sync + ToSig<S>,
        W: ZeroCopy + Word + Send + Sync,
        D: BitFieldSlice<W> + BitFieldSliceMut<W> + Send + Sync,
        S: Sig + ZeroCopy + Send + Sync,
        E: ShardEdge<S, 3>,
        V: ZeroCopy + Send + Sync,
    > VBuilder<T, W, D, S, E, V>
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
        I: IntoIterator<Item = Arc<Vec<SigVal<S, V>>>> + Send,
        C: ConcurrentProgressLog + Send + Sync,
        P: ProgressLog + Clone + Send + Sync,
    >(
        &mut self,
        shard_iter: I,
        data: &mut D,
        new_data: impl Fn(usize, usize) -> D + Sync,
        solve_shard: impl Fn(&Self, usize, &mut Vec<SigVal<S, V>>, D, &mut P) -> Result<(usize, D), ()>
            + Send
            + Sync,
        thread_pool: &rayon::ThreadPool,
        main_pl: &mut C,
        pl: &mut P,
    ) -> Result<(), SolveError>
    where
        I::IntoIter: Send,
        SigVal<S, V>: RadixKey + Send + Sync,
    {
        // TODO: optimize for the non-parallel case
        let (send, receive) =
            crossbeam_channel::bounded::<(usize, D)>(2 * thread_pool.current_num_threads());
        main_pl
            .item_name("shard")
            .expected_updates(Some(self.shard_edge.num_shards()))
            .display_memory(true)
            .start("Solving shards...");
        self.failed.store(false, Ordering::Relaxed);
        let result = thread_pool.scope(|scope| {
            // This thread copies shard-local solutions to the global solution
            scope.spawn(|_| {
                for (shard_index, shard_data) in receive {
                    #[cfg(not(feature = "vbuilder_no_data"))]
                    shard_data.copy(
                        0,
                        data,
                        shard_index * self.shard_edge.num_vertices(),
                        self.shard_edge.num_vertices(),
                    );
                }
            });

            shard_iter
                .into_iter()
                .enumerate()
                .par_bridge()
                .try_for_each_with(
                    (main_pl.clone(), pl.clone()),
                    |(main_pl, pl), (shard_index, shard)| {
                        main_pl.info(format_args!(
                            "Sorting and checking shard {}/{}",
                            shard_index + 1,
                            self.shard_edge.num_shards()
                        ));

                        // Safety: only one thread may be accessing the shard
                        let shard =
                            unsafe { &mut *(Arc::as_ptr(&shard) as *mut Vec<SigVal<S, V>>) };
                        // Check for duplicates
                        shard.radix_sort_builder().with_low_mem_tuner().sort();

                        if shard.par_windows(2).any(|w| w[0].sig == w[1].sig) {
                            return Err(SolveError::DuplicateSignature);
                        }

                        main_pl.info(format_args!(
                            "Solving shard {}/{}",
                            shard_index + 1,
                            self.shard_edge.num_shards()
                        ));

                        if self.failed.load(Ordering::Relaxed) {
                            return Err(SolveError::UnsolvableShard);
                        }

                        solve_shard(
                            self,
                            shard_index,
                            shard,
                            new_data(self.bit_width, self.shard_edge.num_vertices()),
                            pl,
                        )
                        .map_err(|_| {
                            self.failed.store(true, Ordering::Relaxed);
                            SolveError::UnsolvableShard
                        })
                        .map(|(shard_index, data)| {
                            if !self.failed.load(Ordering::Relaxed) {
                                send.send((shard_index, data)).unwrap();
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
                .map(|_| {
                    drop(send);
                })
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
    fn peel_shard<'a, G: Fn(&SigVal<S, V>) -> W + Send + Sync>(
        &self,
        shard_index: usize,
        shard: &'a Vec<SigVal<S, V>>,
        data: D,
        get_val: &G,
        pl: &mut impl ProgressLog,
    ) -> Result<
        (usize, D),
        (
            usize,
            &'a Vec<SigVal<S, V>>,
            Vec<EdgeIndexSideSet>,
            D,
            Vec<usize>,
        ),
    > {
        if self.failed.load(Ordering::Relaxed) {
            return Err((shard_index, shard, vec![], data, vec![]));
        }

        pl.start(format!(
            "Generating graph for shard {}/{}...",
            shard_index + 1,
            self.shard_edge.num_shards()
        ));
        let mut edge_lists = Vec::new();
        edge_lists.resize_with(self.shard_edge.num_vertices(), EdgeIndexSideSet::default);
        shard.iter().enumerate().for_each(|(edge_index, sig_val)| {
            for (side, &v) in self.shard_edge.local_edge(&sig_val.sig).iter().enumerate() {
                edge_lists[v].add(edge_index, side);
            }
        });
        pl.done_with_count(shard.len());

        if self.failed.load(Ordering::Relaxed) {
            return Err((shard_index, shard, edge_lists, data, vec![]));
        }

        pl.start(format!(
            "Peeling graph for shard {}/{}...",
            shard_index + 1,
            self.shard_edge.num_shards()
        ));
        let mut stack = Vec::new();
        // Breadth-first visit in reverse order
        for v in (0..self.shard_edge.num_vertices()).rev() {
            if edge_lists[v].degree() != 1 {
                continue;
            }
            let mut pos = stack.len();
            let mut curr = stack.len();
            stack.push(v);
            while pos < stack.len() {
                let v = stack[pos];
                pos += 1;
                if edge_lists[v].degree() == 0 {
                    continue; // Skip no longer useful entries
                }
                let (edge_index, side) = edge_lists[v].edge_index_and_side();
                edge_lists[v].zero();
                stack[curr] = v;
                curr += 1;

                let e = self.shard_edge.local_edge(&shard[edge_index].sig);
                // Remove edge from the lists of the other two vertices
                match side {
                    0 => {
                        edge_lists[e[1]].remove(edge_index, 1);
                        if edge_lists[e[1]].degree() == 1 {
                            stack.push(e[1]);
                        }
                        edge_lists[e[2]].remove(edge_index, 2);
                        if edge_lists[e[2]].degree() == 1 {
                            stack.push(e[2]);
                        }
                    }
                    1 => {
                        edge_lists[e[0]].remove(edge_index, 0);
                        if edge_lists[e[0]].degree() == 1 {
                            stack.push(e[0]);
                        }
                        edge_lists[e[2]].remove(edge_index, 2);
                        if edge_lists[e[2]].degree() == 1 {
                            stack.push(e[2]);
                        }
                    }
                    2 => {
                        edge_lists[e[0]].remove(edge_index, 0);
                        if edge_lists[e[0]].degree() == 1 {
                            stack.push(e[0]);
                        }
                        edge_lists[e[1]].remove(edge_index, 1);
                        if edge_lists[e[1]].degree() == 1 {
                            stack.push(e[1]);
                        }
                    }
                    _ => unreachable!("{}", side),
                }
            }
            stack.truncate(curr);
        }
        if shard.len() != stack.len() {
            pl.info(format_args!(
                "Peeling failed for shard {}/{} (peeled {} out of {} edges)",
                shard_index + 1,
                self.shard_edge.num_shards(),
                stack.len(),
                shard.len(),
            ));
            return Err((shard_index, shard, edge_lists, data, stack));
        }
        pl.done_with_count(shard.len());

        Ok(self.assign(shard_index, shard, data, get_val, edge_lists, stack, pl))
    }

    /// Perform assignment of values based on peeling data.
    ///
    /// This method might be called after a successful peeling procedure, or
    /// after a linear solver has been used to solve the remaining edges.
    fn assign(
        &self,
        shard_index: usize,
        shard: &Vec<SigVal<S, V>>,
        mut data: D,
        get_val: &(impl Fn(&SigVal<S, V>) -> W + Send + Sync),
        edge_lists: Vec<EdgeIndexSideSet>,
        mut stack: Vec<usize>,
        pl: &mut impl ProgressLog,
    ) -> (usize, D) {
        if self.failed.load(Ordering::Relaxed) {
            return (shard_index, data);
        }

        pl.start(format!(
            "Assigning values for shard {}/{}...",
            shard_index + 1,
            self.shard_edge.num_shards()
        ));
        while let Some(v) = stack.pop() {
            // Assignments after linear solving must skip unpeeled edges
            if edge_lists[v].degree() != 0 {
                continue;
            }
            let (edge_index, side) = edge_lists[v].edge_index_and_side();
            let edge = self.shard_edge.local_edge(&shard[edge_index].sig);
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

        (shard_index, data)
    }

    /// Solve a shard of given index using lazy Gaussian elimination, and store
    /// the solution in the given data.
    ///
    /// Return the shard index and the data, in case of success, or `Err(())` in
    /// case of failure.
    fn lge_shard(
        &self,
        shard_index: usize,
        shard: &Vec<SigVal<S, V>>,
        data: D,
        get_val: &(impl Fn(&SigVal<S, V>) -> W + Send + Sync),
        pl: &mut impl ProgressLog,
    ) -> Result<(usize, D), ()> {
        // Let's try to peel first
        match self.peel_shard(shard_index, shard, data, get_val, pl) {
            Ok((_, data)) => {
                // Unlikely result, but we're happy if it happens
                Ok((shard_index, data))
            }
            Err((shard_index, shard, edge_lists, mut data, stack)) => {
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
                    .filter(|&v| edge_lists[*v].degree() == 0)
                    .for_each(|&v| {
                        unpeeled.set(edge_lists[v].edge_index_and_side().0, false);
                    });
                let unpeeled = Rank9::new(unpeeled);

                // Create data for an F₂ system using non-peeled edges
                let mut var_to_eqs = Vec::with_capacity(self.shard_edge.num_vertices());
                let mut c = vec![W::ZERO; unpeeled.num_ones()];
                var_to_eqs.resize_with(self.shard_edge.num_vertices(), std::vec::Vec::new);
                shard
                    .iter()
                    .enumerate()
                    .filter(|(edge_index, _)| unpeeled[*edge_index])
                    .for_each(|(edge_index, sig_val)| {
                        let eq = unpeeled.rank(edge_index);
                        c[eq] = get_val(sig_val);

                        for &v in self.shard_edge.local_edge(&sig_val.sig).iter() {
                            var_to_eqs[v].push(eq);
                        }
                        ()
                    });
                pl.done_with_count(shard.len());

                if self.failed.load(Ordering::Relaxed) {
                    return Err(());
                }

                pl.expected_updates(Some(unpeeled.num_ones()));
                pl.start("Solving system...");
                let result = Modulo2System::lazy_gaussian_elimination(
                    None,
                    var_to_eqs,
                    c,
                    (0..self.shard_edge.num_vertices()).collect(),
                )
                .map_err(|_| ())?;
                pl.done_with_count(unpeeled.num_ones());

                for (v, &value) in result.iter().enumerate() {
                    data.set(v, value);
                }

                Ok(self.assign(shard_index, shard, data, get_val, edge_lists, stack, pl))
            }
        }
    }
}

/// Build a new function using a vector of `W` to store values.
///
/// Since values are stored in a vector, access is particularly fast, but
/// the bit width of the output of the function is exactly the bit width
/// of `W`.
impl<
        T: ?Sized + Send + Sync + ToSig<S>,
        W: ZeroCopy + Word,
        S: Sig + Send + Sync,
        E: ShardEdge<S, 3>,
    > VBuilder<T, W, Vec<W>, S, E, W>
where
    SigVal<S, W>: RadixKey + Send + Sync,
    Vec<W>: BitFieldSliceMut<W> + BitFieldSlice<W>,
{
    pub fn try_build_func(
        mut self,
        into_keys: impl RewindableIoLender<T>,
        into_values: impl RewindableIoLender<W>,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> anyhow::Result<VFunc<T, W, Vec<W>, S, E>> {
        let get_val = |sig_val: &SigVal<S, W>| sig_val.val;
        let new_data = |_bit_width: usize, len: usize| vec![W::ZERO; len];
        self.build_loop(into_keys, into_values, &get_val, new_data, pl)
    }
}

/// Build a new function using a vector of `W` to store values.
///
/// Since values are stored in a vector, access is particularly fast, but
/// the bit width of the output of the function is exactly the bit width
/// of `W`.
impl<
        T: ?Sized + Send + Sync + ToSig<S>,
        W: ZeroCopy + Word,
        S: Sig + Send + Sync,
        E: ShardEdge<S, 3>,
    > VBuilder<T, W, Vec<W>, S, E, ()>
where
    SigVal<S, ()>: RadixKey + Send + Sync,
    Vec<W>: BitFieldSliceMut<W> + BitFieldSlice<W>,
    u64: CastableInto<W>,
{
    pub fn try_build_filter(
        mut self,
        into_keys: impl RewindableIoLender<T>,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> anyhow::Result<VFilter<W, VFunc<T, W, Vec<W>, S, E>>> {
        let filter_mask = W::MAX;
        let get_val = |sig_val: &SigVal<S, ()>| sig_val.sig.sig_u64().cast();
        let new_data = |_bit_width: usize, len: usize| vec![W::ZERO; len];

        Ok(VFilter {
            func: self.build_loop(
                into_keys,
                FromIntoIterator::from(itertools::repeat_n((), usize::MAX)),
                &get_val,
                new_data,
                pl,
            )?,
            filter_mask,
        })
    }
}

/// Build a new function using a bit-field vector on words of type `W` to store
/// values.
///
/// Since values are stored in a bitfield vector, access will be slower than
/// when using a vector, but the bit width of stored values will be the minimum
/// necessary. It must be in any case less than the bit width of `W`.
///
/// Typically `W` will be `usize` or `u64`. It might be necessary to use
/// `u128` if the bit width of the values is larger than 64.
impl<
        T: ?Sized + Send + Sync + ToSig<S>,
        W: ZeroCopy + Word,
        S: Sig + Send + Sync,
        E: ShardEdge<S, 3>,
    > VBuilder<T, W, BitFieldVec<W>, S, E, W>
where
    SigVal<S, W>: RadixKey + Send + Sync,
{
    pub fn try_build_func(
        mut self,
        into_keys: impl RewindableIoLender<T>,
        into_values: impl RewindableIoLender<W>,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> anyhow::Result<VFunc<T, W, BitFieldVec<W>, S, E>> {
        let get_val = |sig_val: &SigVal<S, W>| sig_val.val;
        let new_data = |bit_width, len| BitFieldVec::<W>::new(bit_width, len);
        self.build_loop(into_keys, into_values, &get_val, new_data, pl)
    }
}

/// Build a new function using a bit-field vector on words of type `W` to store
/// values.
///
/// Since values are stored in a bitfield vector, access will be slower than
/// when using a vector, but the bit width of stored values will be the minimum
/// necessary. It must be in any case less than the bit width of `W`.
///
/// Typically `W` will be `usize` or `u64`. It might be necessary to use
/// `u128` if the bit width of the values is larger than 64.
impl<
        T: ?Sized + Send + Sync + ToSig<S>,
        W: ZeroCopy + Word,
        S: Sig + Send + Sync,
        E: ShardEdge<S, 3>,
    > VBuilder<T, W, BitFieldVec<W>, S, E, ()>
where
    SigVal<S, ()>: RadixKey + Send + Sync,
    Vec<W>: BitFieldSliceMut<W> + BitFieldSlice<W>,
    u64: CastableInto<W>,
{
    pub fn try_build_filter(
        mut self,
        into_keys: impl RewindableIoLender<T>,
        filter_bits: usize,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> anyhow::Result<VFilter<W, VFunc<T, W, BitFieldVec<W>, S, E>>> {
        assert!(filter_bits > 0);
        assert!(filter_bits <= W::BITS);
        let filter_mask = W::MAX >> (W::BITS - filter_bits);
        let get_val = |sig_val: &SigVal<S, ()>| sig_val.sig.sig_u64().cast() & filter_mask;
        let new_data = |bit_width, len| BitFieldVec::<W>::new(bit_width, len);

        Ok(VFilter {
            func: self.build_loop(
                into_keys,
                FromIntoIterator::from(itertools::repeat_n((), usize::MAX)),
                &get_val,
                new_data,
                pl,
            )?,
            filter_mask,
        })
    }
}

impl<
        T: ?Sized + Send + Sync + ToSig<S>,
        W: ZeroCopy + Word,
        D: BitFieldSlice<W> + BitFieldSliceMut<W> + Send + Sync,
        S: Sig + Send + Sync,
        E: ShardEdge<S, 3>,
        V: ZeroCopy + Send + Sync,
    > VBuilder<T, W, D, S, E, V>
where
    SigVal<S, V>: RadixKey + Send + Sync,
{
    /*fn set_up_shards(&mut self, num_keys: usize) {
        let eps = 0.001; // Tolerance for deviation from the average shard size
        self.shard_high_bits = if SHARDED {
            if num_keys <= MAX_LIN_SIZE {
                // We just try to make shards as big as possible,
                // within a maximum size of 2 * MAX_LIN_SHARD_SIZE
                (num_keys / MAX_LIN_SHARD_SIZE).max(1).ilog2()
            } else {
                // Bound from urns and balls problem
                let t = (num_keys as f64 * eps * eps / 2.0).ln();

                if t > 0.0 {
                    // We correct the estimate to increase slightly the shard size
                    ((t - 1.92 * t.ln() - 1.22 * t.ln().ln()) / 2_f64.ln())
                        .ceil()
                        .max(3.) as u32
                } else {
                    0
                }
                .min(LOG2_MAX_SHARDS) // We don't really need too many shards
                .min((num_keys / MIN_FUSE_SHARD).max(1).ilog2()) // Shards can't smaller than MIN_FUSE_SHARD
            }
        } else {
            0
        };

        self.shard_edge.num_shards = 1 << self.shard_high_bits;
        self.shard_mask = (1u32 << self.shard_high_bits) - 1;
    }

    fn set_up_segments(&mut self) {
        let shard_size = self.num_keys.div_ceil(self.shard_edge.num_shards);
        if SHARDED {
            self.lazy_gaussian = self.num_keys <= MAX_LIN_SIZE;

            (self.c, self.log2_seg_size) = if self.lazy_gaussian {
                (1.10, lin_log2_seg_size(3, shard_size))
            } else {
                (1.105, fuse_log2_seg_size(3, shard_size))
            };
        } else {
            self.lazy_gaussian = self.num_keys <= MAX_LIN_SHARD_SIZE;

            (self.c, self.log2_seg_size) = if self.lazy_gaussian {
                (1.10, lin_log2_seg_size(3, self.num_keys))
            } else if self.num_keys <= MAX_LIN_SIZE {
                (
                    fuse_c(3, self.num_keys),
                    fuse_log2_seg_size(3, self.num_keys),
                )
            } else {
                (1.105, fuse_log2_seg_size(3, self.num_keys) + 1)
            };
        }

        self.l = ((self.c * shard_size as f64).ceil() as usize).div_ceil(1 << self.log2_seg_size);
        self.shard_edge.num_vertices() = (1 << self.log2_seg_size) * (self.l + 2);
    }

    */
}

impl<
        T: ?Sized + Send + Sync + ToSig<S>,
        W: ZeroCopy + Word,
        D: BitFieldSlice<W> + BitFieldSliceMut<W> + Send + Sync,
        S: Sig + Send + Sync,
        E: ShardEdge<S, 3>,
        V: ZeroCopy + Send + Sync,
    > VBuilder<T, W, D, S, E, V>
where
    SigVal<S, V>: RadixKey + Send + Sync,
{
    /// Build and return a new function with given keys and values.
    ///
    /// This function can build functions based both on vectors and on bit-field
    /// vectors. The necessary abstraction is provided by the `new(bit_width,
    /// len)` function, which is called to create the data structure to store
    /// the values.
    fn build_loop(
        &mut self,
        mut into_keys: impl RewindableIoLender<T>,
        mut into_values: impl RewindableIoLender<V>,
        get_val: &(impl Fn(&SigVal<S, V>) -> W + Send + Sync),
        new: fn(usize, usize) -> D,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> anyhow::Result<VFunc<T, W, D, S, E>> {
        let mut dup_count = 0;
        let start = Instant::now();
        let mut prng = SmallRng::seed_from_u64(self.seed);

        // Loop until success or duplicate detection
        loop {
            let seed = prng.random();
            pl.item_name("key");
            pl.start("Reading input...");

            into_values = into_values.rewind()?;
            into_keys = into_keys.rewind()?;

            match if self.offline {
                self.try_seed(
                    seed,
                    sig_store::new_offline::<S, V>(self.log2_buckets, LOG2_MAX_SHARDS)?,
                    &mut into_keys,
                    &mut into_values,
                    get_val,
                    new,
                    pl,
                )
            } else {
                self.try_seed(
                    seed,
                    sig_store::new_online::<S, V>(self.log2_buckets, LOG2_MAX_SHARDS)?,
                    &mut into_keys,
                    &mut into_values,
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
        T: ?Sized + Send + Sync + ToSig<S>,
        W: ZeroCopy + Word,
        D: BitFieldSlice<W> + BitFieldSliceMut<W> + Send + Sync,
        S: Sig + Send + Sync,
        E: ShardEdge<S, 3>,
        V: ZeroCopy + Send + Sync,
    > VBuilder<T, W, D, S, E, V>
where
    SigVal<S, V>: RadixKey + Send + Sync,
{
    fn try_seed<G: Fn(&SigVal<S, V>) -> W + Send + Sync>(
        &mut self,
        seed: u64,
        mut sig_store: impl SigStore<S, V>,
        into_keys: &mut impl RewindableIoLender<T>,
        into_values: &mut impl RewindableIoLender<V>,
        get_val: &G,
        new_data: fn(usize, usize) -> D,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> anyhow::Result<VFunc<T, W, D, S, E>> {
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

        while let Some(result) = into_keys.next() {
            match result {
                Ok(key) => {
                    pl.light_update();
                    // This might be an actual value, if we are building a
                    // function, or (), if we are building a filter.
                    let &maybe_val = into_values.next().expect("Not enough values")?;
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

        (self.c, self.lazy_gaussian) = self.shard_edge.set_up_shards(self.num_keys);

        let mut shard_store = sig_store.into_shard_store(self.shard_edge.shard_high_bits())?;
        let max_shard = shard_store.shard_sizes().iter().copied().max().unwrap_or(0);

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
            Err(Into::into(SolveError::MaxShardTooBig))
        } else {
            #[cfg(not(feature = "vbuilder_no_data"))]
            let data = new_data(
                self.bit_width,
                self.shard_edge.num_vertices() * self.shard_edge.num_shards(),
            );
            #[cfg(feature = "vbuilder_no_data")]
            let data = new_data(self.bit_width, 0);
            self.try_build_from_shard_iter(seed, data, shard_store.iter(), new_data, get_val, pl)
                .map_err(Into::into)
        }
    }

    /// Build and return a new function starting from an iterator on shards.
    ///
    /// This method provide construction logic that is independent from the
    /// actual storage of the values (offline or in core memory.)
    ///
    /// See [`VBuilder::_build`] for more details on the parameters.
    fn try_build_from_shard_iter<I, P, G: Fn(&SigVal<S, V>) -> W + Send + Sync>(
        &mut self,
        seed: u64,
        mut data: D,
        shard_iter: I,
        new: fn(usize, usize) -> D,
        get_val: &G,
        pl: &mut P,
    ) -> Result<VFunc<T, W, D, S, E>, SolveError>
    where
        P: ProgressLog + Clone + Send + Sync,
        I: Iterator<Item = Arc<Vec<SigVal<S, V>>>> + Send,
    {
        let thread_pool = ThreadPoolBuilder::new()
            .num_threads(self.shard_edge.num_shards().min(self.max_num_threads) + 1) // Or it might hang
            .build()
            .unwrap(); // Seroiusly, it's not going to fail

        pl.info(format_args!("{}", self.shard_edge));
        pl.info(format_args!(
            "c: {}, Overhead: {:.2}% Number of threads: {}",
            self.c,
            (self.shard_edge.num_vertices() * self.shard_edge.num_shards()) as f64
                / (self.num_keys as f64),
            thread_pool.current_num_threads()
        ));

        self.par_solve(
            shard_iter,
            &mut data,
            new,
            |this, shard_index, shard, data, pl| {
                this.lge_shard(shard_index, shard, data, get_val, pl)
            },
            &thread_pool,
            &mut pl.concurrent(),
            pl,
        )?;

        pl.info(format_args!(
            "Bits/keys: {} ({:.2}%)",
            data.len() as f64 * self.bit_width as f64 / self.num_keys as f64,
            100.0 * data.len() as f64 / self.num_keys as f64,
        ));

        Ok(VFunc::<T, W, D, S, E> {
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
