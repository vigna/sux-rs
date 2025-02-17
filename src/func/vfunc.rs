/*
*
* SPDX-FileCopyrightText: 2023 Sebastiano Vigna
*
* SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
*/

// We import selectively to be able to use AtomicHelper
use crate::bits::*;
use crate::prelude::Rank9;
use crate::traits::bit_field_slice;
use crate::traits::NumBits;
use crate::traits::Rank;
use crate::utils::*;
use anyhow::bail;
use arbitrary_chunks::ArbitraryChunkMut;
use arbitrary_chunks::ArbitraryChunks;
use bit_field_slice::BitFieldSlice;
use bit_field_slice::BitFieldSliceMut;
use bit_field_slice::Word;
use derivative::Derivative;
use derive_setters::*;
use dsi_progress_logger::*;
use epserde::prelude::*;
use mem_dbg::*;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use rdst::*;
use std::borrow::BorrowMut;
use std::marker::PhantomData;
use std::time::Instant;

fn lin_log2_seg_size(arity: usize, n: usize) -> u32 {
    match arity {
        3 => {
            if n >= 100_000 {
                9
            } else {
                (0.76 * (n.max(1) as f64).ln()).floor() as u32
            }
        }
        _ => unimplemented!(),
    }
}

fn fuse_log2_seg_size(arity: usize, n: usize) -> u32 {
    // From “Binary Fuse Filters: Fast and Smaller Than Xor Filters”
    // https://doi.org/10.1145/3510449
    match arity {
        3 => ((n.max(1) as f64).ln() / (3.33_f64).ln() + 2.25).floor() as u32,
        4 => ((n.max(1) as f64).ln() / (2.91_f64).ln() - 0.5).floor() as u32,
        _ => unimplemented!(),
    }
}

fn fuse_c(arity: usize, n: usize) -> f64 {
    // From “Binary Fuse Filters: Fast and Smaller Than Xor Filters”
    // https://doi.org/10.1145/3510449
    match arity {
        3 => 1.125_f64.max(0.875 + 0.25 * (1000000_f64).ln() / (n as f64).ln()),
        4 => 1.075_f64.max(0.77 + 0.305 * (600000_f64).ln() / (n as f64).ln()),
        _ => unimplemented!(),
    }
}

fn count_sort<T: Copy, A: AsMut<[T]>, K: Fn(T) -> usize>(
    mut data: A,
    num_keys: usize,
    key: K,
) -> Vec<usize> {
    let data = data.as_mut();

    let mut counts = vec![0; num_keys];
    for &x in &*data {
        counts[key(x)] += 1;
    }

    let mut last_used = 0;
    let mut pos = vec![0; num_keys];
    let mut p = 0;
    // Compute cumulative distribution
    for (i, &c) in counts.iter().enumerate() {
        if c != 0 {
            last_used = i;
        }
        p += c;
        pos[i] = p;
    }

    // Skip empty buckets
    let end = data.len() - counts[last_used];

    let counts_clone = counts.clone();
    let mut i = 0;

    // Dutch national flag algorithm
    while i <= end {
        let mut t = data[i];
        let mut c = key(t);
        if i < end {
            loop {
                pos[c] -= 1;

                if pos[c] <= i {
                    break;
                }

                std::mem::swap(&mut t, &mut data[pos[c]]);
                c = key(t);
            }

            data[i] = t;
        }

        i += counts[c];
        counts[c] = 0;
    }

    counts_clone
}

/// An edge list represented by a 64-bit integer. The lower SIDE_SHIFT bits
/// contain a XOR of the edges; the next two bits contain a XOR of the side of
/// the edges; the remaining DEG_SHIFT the upper bits contain the degree.
///
/// The degree can be stored with a small number of bits because the graph is
/// random, so the maximum degree is O(log log n).
///
/// This approach reduces the core memory usage for the hypergraph to a `u64`
/// per vertex. Edges are derived on the fly from the 128-bit signatures.
#[derive(Debug, Default)]
struct EdgeList(u64);
impl EdgeList {
    const DEG_SHIFT: u32 = u64::BITS as u32 - 16;
    const SIDE_SHIFT: u32 = u64::BITS as u32 - 18;
    const SIDE_MASK: usize = 3;
    const EDGE_INDEX_MASK: u64 = (1_u64 << Self::SIDE_SHIFT) - 1;
    const DEG: u64 = 1_u64 << Self::DEG_SHIFT;
    const MAX_DEG: usize = (u64::MAX >> Self::DEG_SHIFT) as usize;

    #[inline(always)]
    fn add(&mut self, edge: usize, side: usize) {
        debug_assert!(self.degree() < Self::MAX_DEG);
        self.0 += Self::DEG;
        self.0 ^= (side as u64) << Self::SIDE_SHIFT | edge as u64;
    }

    #[inline(always)]
    fn remove(&mut self, edge: usize, side: usize) {
        debug_assert!(self.degree() > 0);
        self.0 -= Self::DEG;
        self.0 ^= (side as u64) << Self::SIDE_SHIFT | edge as u64;
    }

    /// The degree of the vertex.
    ///
    /// When the degree is zero, the entry is used to store a [peeled edge and
    /// its hinge](EdgeList::edge_index_and_hinge).
    #[inline(always)]
    fn degree(&self) -> usize {
        (self.0 >> Self::DEG_SHIFT) as usize
    }

    /// Retrieve a peeled edge index and its hinge (as an index in the edge's
    /// vertices).
    ///
    /// This method return meaningful values only when the [degree is
    /// zero](EdgeList::degree).
    #[inline(always)]
    fn edge_index_and_hinge(&self) -> (usize, usize) {
        // debug_assert!(self.degree() == 0); TODO
        (
            (self.0 & EdgeList::EDGE_INDEX_MASK) as usize,
            (self.0 >> Self::SIDE_SHIFT) as usize & Self::SIDE_MASK,
        )
    }

    #[inline(always)]
    fn zero(&mut self) {
        self.0 &= (1 << Self::DEG_SHIFT) - 1;
    }
}

/*

Chunk and edge information is derived from the 128-bit signatures of the keys.
More precisely, each signature is made of two 64-bit integers `h` and `l`, and then:

- the `high_bits` most significant bits of `h` are used to select a shard;
- the `log2_l` least significant bits of the upper 32 bits of `h` are used to select a segment;
- the lower 32 bits of `h` are used to select the virst vertex;
- the upper 32 bits of `l` are used to select the second vertex;
- the lower 32 bits of `l` are used to select the third vertex.

*/

/// Static functions with 10%-11% space overhead for large key sets,
/// fast parallel construction, and fast queries.
///
/// Instances of this structure are immutable; they are built using a [`VFuncBuilder`],
/// and can be serialized using [ε-serde](`epserde`).
#[derive(Epserde, Debug, MemDbg, MemSize)]
pub struct VFunc<
    T: ?Sized + ToSig,
    W: ZeroCopy + Word = usize,
    D: bit_field_slice::BitFieldSlice<W> = BitFieldVec<W>,
    S = [u64; 2],
    const SHARDED: bool = false,
> {
    shard_high_bits: u32,
    log2_seg_size: u32,
    shard_mask: u32,
    seed: u64,
    l: usize,
    num_keys: usize,
    data: D,
    _marker_t: std::marker::PhantomData<T>,
    _marker_os: std::marker::PhantomData<(W, S)>,
}

/// A builder for [`VFunc`].
///
/// Keys must implement the [`ToSig`] trait, which provides a method to compute
/// a 128-bit signature of the key.
///
/// The output type `O` can be selected to be any of the unsigned integer types;
/// the default is `usize`.
///
/// There are two construction modes: in core memory (default) and
/// [offline](VFuncBuilder::offline). In the first case, space will be allocated
/// in core memory for 128-bit signatures and associated values for all keys; in
/// the second case, such information will be stored in a number of on-disk
/// buckets.
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
pub struct VFuncBuilder<
    T: ?Sized + Send + Sync + ToSig,
    W: ZeroCopy + Word,
    D: bit_field_slice::BitFieldSlice<W> + Send + Sync = BitFieldVec<W>,
    S = [u64; 2],
    const SHARDED: bool = false,
> {
    /// The (optional) expected number of keys.s
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

    /// The base-2 logarithm of the number of on-disk buckets. Used only if
    /// `offline` is `true`. The default is 8.
    #[setters(generate = true, strip_option)]
    #[derivative(Default(value = "8"))]
    log2_buckets: u32,

    /// The bit width of the maximum value.
    bit_width: usize,
    /// The number of high bits defining a shard. It is zero if `SHARDED` is
    /// false.
    shard_high_bits: u32,
    /// The mask to select a shard.
    shard_mask: u32,
    /// The number of keys.
    num_keys: usize,
    /// The number of shards, i.e., `(1 << shard_high_bits)`.
    num_shards: usize,
    /// The base-2 logarithm of the segment size.
    log2_seg_size: u32,
    /// The number of segments minus 2.
    l: usize,
    /// The number of vertices in a hypergraph, i.e., `(1 << log2_seg_size) * (l + 2)`.
    num_vertices: usize,

    #[doc(hidden)]
    _marker_t: PhantomData<T>,
    #[doc(hidden)]
    _marker_od: PhantomData<(W, D, S)>,
}

#[derive(thiserror::Error, Debug)]
/// Errors that can happen during deserialization.
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
        T: ?Sized + Send + Sync + ToSig,
        W: ZeroCopy + Word + Send + Sync,
        D: BitFieldSlice<W> + BitFieldSliceMut<W> + Send + Sync,
        S: Send + Sync,
        const SHARDED: bool,
    > VFuncBuilder<T, W, D, S, SHARDED>
{
    /// Solve in parallel the shards returned by the given iterator.
    ///
    /// # Arguments
    ///
    /// * `main_pl`: the progress logger for the overall computation.
    ///
    /// * `pl`: a progress logger that will be cloned to display the progress of
    ///   a current shard.
    ///
    /// * `shard_iter`: an iterator returning the shards to solve.
    ///
    /// * `data`: the storage for the solution values.
    ///
    /// * `new`: a function to create shard-local storage for the values.
    ///
    /// * `do_shard`: a method to solve a shard.
    ///
    /// # Errors
    ///
    /// This method returns an error if one of the shards cannot be solved,
    /// or if duplicates are detected.
    fn par_solve<
        'a,
        I: IntoIterator<Item = (usize, B)> + Send,
        B: BorrowMut<[SigVal<W>]> + Send,
        C: ConcurrentProgressLog + Send + Sync,
        P: ProgressLog + Clone + Send + Sync,
    >(
        &mut self,
        shard_iter: I,
        data: &mut D,
        new_data: impl Fn(usize, usize) -> D + Sync,
        solve_shard: impl Fn(&Self, usize, B, D, &mut P) -> Result<(usize, D), ()> + Send + Sync,
        thread_pool: &rayon::ThreadPool,
        main_pl: &mut C,
        pl: &mut P,
    ) -> Result<(), SolveError>
    where
        I::IntoIter: Send,
        SigVal<W>: RadixKey + Send + Sync,
    {
        // TODO: optimize for the non-parallel case
        let (send, receive) =
            crossbeam_channel::bounded::<(usize, D)>(2 * thread_pool.current_num_threads());
        main_pl
            .item_name("shard")
            .expected_updates(Some(self.num_shards))
            .display_memory(true)
            .start("Solving shards...");
        let result = thread_pool.scope(|scope| {
            scope.spawn(|_| {
                for (shard_index, shard_data) in receive {
                    // TODO: fast copy
                    for i in 0..self.num_vertices {
                        data.set(shard_index * self.num_vertices + i, shard_data.get(i));
                    }
                }
            });

            shard_iter
                .into_iter()
                .par_bridge()
                .try_for_each_with(
                    (main_pl.clone(), pl.clone()),
                    |(main_pl, pl), (shard_index, mut shard)| {
                        main_pl.info(format_args!(
                            "Sorting and checking shard {}/{}",
                            shard_index + 1,
                            self.num_shards
                        ));

                        shard.borrow_mut().radix_sort_unstable();

                        if shard
                            .borrow_mut()
                            .par_windows(2)
                            .any(|w| w[0].sig == w[1].sig)
                        {
                            return Err(SolveError::DuplicateSignature);
                        }

                        main_pl.info(format_args!(
                            "Solving shard {}/{}",
                            shard_index + 1,
                            self.num_shards
                        ));

                        solve_shard(
                            self,
                            shard_index,
                            shard,
                            new_data(self.bit_width, self.num_vertices),
                            pl,
                        )
                        .map_err(|_| SolveError::UnsolvableShard)
                        .map(|(shard_index, data)| {
                            send.send((shard_index, data)).unwrap();
                            main_pl.info(format_args!(
                                "Completed shard {}/{}",
                                shard_index + 1,
                                self.num_shards
                            ));
                            main_pl.update_and_display();
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
    /// Return the shard index and the data, in case of success,
    /// or the shard index, the shard, the edge lists, the data, and the stack
    /// at the end of the peeling procedure in case of failure. These data can
    /// be then passed to a linear solver to complete the solution.
    fn peel_shard<'a, B: BorrowMut<[SigVal<W>]>>(
        &self,
        shard_index: usize,
        shard: B,
        data: D,
        pl: &mut impl ProgressLog,
    ) -> Result<(usize, D), (usize, B, Vec<EdgeList>, D, Vec<usize>)> {
        pl.start(format!(
            "Generating graph for shard {}/{}...",
            shard_index + 1,
            self.num_shards
        ));
        let mut edge_lists = Vec::new();
        edge_lists.resize_with(self.num_vertices, EdgeList::default);
        shard
            .borrow()
            .iter()
            .enumerate()
            .for_each(|(edge_index, sig_val)| {
                for (side, &v) in sig_val
                    .sig
                    .edge(self.shard_high_bits, self.l, self.log2_seg_size)
                    .iter()
                    .enumerate()
                {
                    edge_lists[v].add(edge_index, side);
                }
            });
        pl.done_with_count(shard.borrow().len());

        pl.start(format!(
            "Peeling graph for shard {}/{}...",
            shard_index + 1,
            self.num_shards
        ));
        let mut stack = Vec::new();
        for v in (0..self.num_vertices).rev() {
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
                let (edge_index, side) = edge_lists[v].edge_index_and_hinge();
                edge_lists[v].zero();
                stack[curr] = v;
                curr += 1;

                let e = shard.borrow()[edge_index].sig.edge(
                    self.shard_high_bits,
                    self.l,
                    self.log2_seg_size,
                );
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
        if shard.borrow().len() != stack.len() {
            pl.info(format_args!(
                "Peeling failed for shard {}/{} (peeled {} out of {} edges)",
                shard_index + 1,
                self.num_shards,
                stack.len(),
                shard.borrow().len(),
            ));
            return Err((shard_index, shard, edge_lists, data, stack));
        }
        pl.done_with_count(shard.borrow().len());

        Ok(self.assign(shard_index, shard, data, edge_lists, stack, pl))
    }

    /// Perform assignment of values based on peeling data.
    ///
    /// This method might be called after a successful peeling procedure,
    /// or after a linear solver has been used to solve the remaining edges.
    fn assign(
        &self,
        shard_index: usize,
        shard: impl BorrowMut<[SigVal<W>]>,
        mut data: D,
        edge_lists: Vec<EdgeList>,
        mut stack: Vec<usize>,
        pl: &mut impl ProgressLog,
    ) -> (usize, D) {
        let shard = shard.borrow();
        pl.start(format!(
            "Assigning values for shard {}/{}...",
            shard_index + 1,
            self.num_shards
        ));
        while let Some(v) = stack.pop() {
            // Assignments after linear solving must skip unpeeled edges
            if edge_lists[v].degree() != 0 {
                continue;
            }
            let (edge_index, side) = edge_lists[v].edge_index_and_hinge();
            let edge = shard[edge_index]
                .sig
                .edge(self.shard_high_bits, self.l, self.log2_seg_size);
            let value = match side {
                0 => data.get(edge[1]) ^ data.get(edge[2]),
                1 => data.get(edge[0]) ^ data.get(edge[2]),
                2 => data.get(edge[0]) ^ data.get(edge[1]),
                _ => unreachable!(),
            };

            data.set(v, shard[edge_index].val ^ value);
            debug_assert_eq!(
                data.get(edge[0]) ^ data.get(edge[1]) ^ data.get(edge[2]),
                shard[edge_index].val
            );
        }
        pl.done_with_count(shard.len());

        (shard_index, data)
    }

    /// Solve a shard of given index using lazy Gaussian elimination,
    /// and store the solution in the given data.
    ///
    /// Return the shard index and the data, in case of success,
    /// or `Err(())` in case of failure.
    fn solve_lin<'a>(
        &self,
        shard_index: usize,
        shard: impl BorrowMut<[SigVal<W>]>,
        data: D,
        pl: &mut impl ProgressLog,
    ) -> Result<(usize, D), ()> {
        // Let's try to peel first
        match self.peel_shard(shard_index, shard, data, pl) {
            Ok((_, data)) => {
                // Unlikely result, but we're happy if it happens
                return Ok((shard_index, data));
            }
            Err((shard_index, shard, edge_lists, mut data, stack)) => {
                // Likely result--we have solve the rest
                pl.start(format!(
                    "Generating system for shard {}/{}...",
                    shard_index + 1,
                    self.num_shards
                ));

                // Build a ranked vector of unpeeled equation
                let mut unpeeled = bit_vec![true; shard.borrow().len()];
                stack
                    .iter()
                    .filter(|&v| edge_lists[*v].degree() == 0)
                    .for_each(|&v| {
                        unpeeled.set(edge_lists[v].edge_index_and_hinge().0, false);
                    });
                let unpeeled = Rank9::new(unpeeled);

                // Create data for an F₂ system using non-peeled edges
                let mut var_to_eqs = Vec::with_capacity(self.num_vertices);
                let mut c = vec![W::ZERO; unpeeled.num_ones()];
                var_to_eqs.resize_with(self.num_vertices, || Vec::with_capacity(16));
                shard
                    .borrow()
                    .iter()
                    .enumerate()
                    .filter(|(edge_index, _)| unpeeled[*edge_index])
                    .for_each(|(edge_index, sig_val)| {
                        let eq = unpeeled.rank(edge_index);
                        c[eq] = sig_val.val;

                        for &v in sig_val
                            .sig
                            .edge(self.shard_high_bits, self.l, self.log2_seg_size)
                            .iter()
                        {
                            var_to_eqs[v].push(eq);
                        }
                    });
                pl.done_with_count(shard.borrow().len());

                pl.start("Solving system...");
                let result = Modulo2System::lazy_gaussian_elimination(
                    None,
                    var_to_eqs,
                    c,
                    (0..self.num_vertices).collect(),
                )
                .map_err(|_| ())?;
                pl.done();

                for (v, &value) in result.iter().enumerate() {
                    data.set(v, value);
                }

                Ok(self.assign(shard_index, shard, data, edge_lists, stack, pl))
            }
        }
    }
}

pub trait ChunkEdge: Send + Sync {
    fn shard(&self, shard_high_bits: u32, shard_mask: u32) -> usize;
    fn edge(&self, shard_high_bits: u32, l: usize, log2_seg_size: u32) -> [usize; 3];
}

impl ChunkEdge for [u64; 2] {
    #[inline(always)]
    #[must_use]
    fn shard(&self, shard_high_bits: u32, shard_mask: u32) -> usize {
        // TODO: shift?
        (self[0].rotate_left(shard_high_bits) & shard_mask as u64) as usize
    }

    #[inline(always)]
    #[must_use]
    fn edge(&self, shard_high_bits: u32, l: usize, log2_seg_size: u32) -> [usize; 3] {
        let first_segment = ((self[0] << shard_high_bits >> 32) * l as u64 >> 32) as usize;
        let start = first_segment << log2_seg_size;
        let segment_size = 1 << log2_seg_size;
        let segment_mask = segment_size - 1;

        [
            (self[0] as usize & segment_mask) + start,
            ((self[1] >> 32) as usize & segment_mask) + start + segment_size,
            (self[1] as usize & segment_mask) + start + 2 * segment_size,
        ]
    }
}

impl ChunkEdge for u64 {
    fn shard(&self, shard_high_bits: u32, shard_mask: u32) -> usize {
        let xor = *self as u32 ^ (*self >> 32) as u32;
        (xor.rotate_left(shard_high_bits) & shard_mask) as usize
    }

    #[inline(always)]
    #[must_use]
    fn edge(&self, shard_high_bits: u32, l: usize, log2_seg_size: u32) -> [usize; 3] {
        let xor = *self as u32 ^ (*self >> 32) as u32;
        let first_segment = ((xor.rotate_left(shard_high_bits) as u64 * l as u64) >> 32) as usize;
        let start = first_segment << log2_seg_size;
        let segment_size = 1 << log2_seg_size;
        let segment_mask = segment_size - 1;

        [
            (*self as usize & segment_mask) as usize + start,
            ((*self >> 21) as usize & segment_mask) + start + segment_size,
            ((*self >> 42) as usize & segment_mask) + start + 2 * segment_size,
        ]
    }
}

impl<
        T: ?Sized + ToSig,
        W: ZeroCopy + Word,
        D: bit_field_slice::BitFieldSlice<W>,
        S: ChunkEdge,
        const SHARDED: bool,
    > VFunc<T, W, D, S, SHARDED>
{
    /// Return the value associated with the given signature.
    ///
    /// This method is mainly useful in the construction of compound functions.
    #[inline(always)]
    pub fn get_by_sig(&self, sig: &[u64; 2]) -> W {
        if SHARDED {
            let edge = sig.edge(self.shard_high_bits, self.l, self.log2_seg_size);
            let chunk = sig.shard(self.shard_high_bits, self.shard_mask);
            // chunk * self.segment_size * (2^log2_l + 2)
            let shard_offset = chunk * ((self.l + 2) << self.log2_seg_size);

            unsafe {
                self.data.get_unchecked(edge[0] + shard_offset)
                    ^ self.data.get_unchecked(edge[1] + shard_offset)
                    ^ self.data.get_unchecked(edge[2] + shard_offset)
            }
        } else {
            let edge = sig.edge(0, self.l, self.log2_seg_size);
            unsafe {
                self.data.get_unchecked(edge[0])
                    ^ self.data.get_unchecked(edge[1])
                    ^ self.data.get_unchecked(edge[2])
            }
        }
    }

    /// Return the value associated with the given key, or a random value if the key is not present.
    #[inline(always)]
    pub fn get(&self, key: &T) -> W {
        self.get_by_sig(&T::to_sig(key, self.seed))
    }

    /// Return the number of keys in the function.
    pub fn len(&self) -> usize {
        self.num_keys
    }

    /// Return whether the function has no keys.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Build a new function using a vector of `W` to store values.
///
/// Since values are stored in a vector, access is particularly fast, but
/// the bit width of the output of the function is exactly the bit width
/// of `W`.
impl<T: ?Sized + Send + Sync + ToSig, W: ZeroCopy + Word, S: ChunkEdge, const SHARDED: bool>
    VFuncBuilder<T, W, Vec<W>, S, SHARDED>
where
    SigVal<W>: RadixKey + Send + Sync,
    Vec<W>: BitFieldSliceMut<W> + BitFieldSlice<W>,
{
    pub fn build(
        mut self,
        into_keys: impl RewindableIoLender<T>,
        into_values: impl RewindableIoLender<W>,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> anyhow::Result<VFunc<T, W, Vec<W>, S, SHARDED>> {
        self._build(
            into_keys,
            into_values,
            |_bit_width: usize, len: usize| vec![W::ZERO; len],
            pl,
        )
    }
}

struct IntoChunkIter<T> {
    data: Vec<T>,
    shard_sizes: Vec<usize>,
}

impl<'a, T: Copy + 'static> IntoIterator for &'a mut IntoChunkIter<T> {
    type IntoIter = ArbitraryChunkMut<'a, 'a, T>;
    type Item = &'a mut [T];

    fn into_iter(self) -> Self::IntoIter {
        self.data.arbitrary_chunks_mut(&self.shard_sizes)
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
impl<T: ?Sized + Send + Sync + ToSig, W: ZeroCopy + Word, S: ChunkEdge, const SHARDED: bool>
    VFuncBuilder<T, W, BitFieldVec<W>, S, SHARDED>
where
    SigVal<W>: RadixKey + Send + Sync,
{
    pub fn build(
        mut self,
        into_keys: impl RewindableIoLender<T>,
        into_values: impl RewindableIoLender<W>,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> anyhow::Result<VFunc<T, W, BitFieldVec<W>, S, SHARDED>> {
        self._build(
            into_keys,
            into_values,
            |bit_width: usize, len: usize| BitFieldVec::<W>::new(bit_width, len),
            pl,
        )
    }
}

const MAX_LIN_SIZE: usize = 2_000_000;
const MAX_LIN_SHARD_SIZE: usize = 100_000;
const LOG2_MAX_SHARDS: u32 = 8;

impl<
        T: ?Sized + Send + Sync + ToSig,
        W: ZeroCopy + Word,
        D: bit_field_slice::BitFieldSlice<W> + bit_field_slice::BitFieldSliceMut<W> + Send + Sync,
        S: ChunkEdge,
        const SHARDED: bool,
    > VFuncBuilder<T, W, D, S, SHARDED>
where
    SigVal<W>: RadixKey + Send + Sync,
{
    /// Return the number of high bits defining shards.
    fn set_up_shards(&mut self) {
        let eps = 0.001; // Tolerance for deviation from the average shard size
        let num_keys = self.num_keys;
        self.shard_high_bits = if SHARDED {
            if num_keys <= MAX_LIN_SIZE {
                // We just try to make shards as big as possible,
                // within a maximum size of 2 * MAX_LIN_SHARD_SIZE
                (num_keys / MAX_LIN_SHARD_SIZE).max(1).ilog2()
            } else {
                // Bound from urns and balls problem
                let t = (num_keys as f64 * eps * eps / 2.0).ln();

                if t > 0.0 {
                    ((t - t.ln()) / 2_f64.ln()).ceil() as u32
                } else {
                    0
                }
                .min(LOG2_MAX_SHARDS) // We don't really need too many shards
                .min((num_keys / MAX_LIN_SIZE).max(1).ilog2()) // Shards can't smaller than MAX_LIN_SIZE
            }
        } else {
            0
        };

        self.num_shards = 1 << self.shard_high_bits;
        self.shard_mask = (1u32 << self.shard_high_bits) - 1;
    }

    /// Build and return a new function with given keys and values.
    ///
    /// This function can build functions based both on vectors and on bit-field
    /// vectors. The necessary abstraction is provided by the `new(bit_width,
    /// len)` function, which is called to create the data structure to store
    /// the values.
    fn _build(
        &mut self,
        mut into_keys: impl RewindableIoLender<T>,
        mut into_values: impl RewindableIoLender<W>,
        new: fn(usize, usize) -> D,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> anyhow::Result<VFunc<T, W, D, S, SHARDED>> {
        let (mut seed, mut dup_count) = (0, 0);
        let start = Instant::now();

        // Loop until success or duplicate detection
        loop {
            pl.item_name("key");
            pl.start("Reading input...");
            let mut max_value = W::ZERO;

            match if self.offline {
                // Offline construction using a SigStore
                pl.info(format_args!("Using {} buckets", 1 << self.log2_buckets));
                // Put values into a signature sorter
                let mut sig_sorter =
                    SigStore::<W>::new(self.log2_buckets, LOG2_MAX_SHARDS).unwrap();
                into_values = into_values.rewind()?;
                into_keys = into_keys.rewind()?;
                while let Some(result) = into_keys.next() {
                    match result {
                        Ok(key) => {
                            pl.light_update();
                            let &v = into_values.next().expect("Not enough values")?;
                            max_value = Ord::max(max_value, v);
                            sig_sorter.push(&SigVal {
                                sig: T::to_sig(key, seed),
                                val: v,
                            })?;
                        }
                        Err(e) => {
                            bail!("Error reading input: {}", e);
                        }
                    }
                }
                pl.done();

                self.num_keys = sig_sorter.len();
                self.bit_width = max_value.len() as usize;

                self.set_up_shards();

                let shard_store = sig_sorter.into_chunk_store(self.shard_high_bits)?;
                let max_shard = shard_store.chunk_sizes().iter().copied().max().unwrap_or(0);
                if max_shard as f64 > 1.001 * self.num_keys as f64 / self.num_shards as f64 {
                    bail!("Shards are too small");
                }

                pl.info(format_args!(
                    "Keys: {} Max value: {} Bit width: {} Shards: 2^{} Max shard / average shard: {:.2}%",
                    self.num_keys, max_value, self.bit_width, self.shard_high_bits,
                    (100.0 * max_shard as f64) / (self.num_keys as f64 / self.num_shards as f64)
                ));

                self.build_iter(seed, shard_store, max_shard, new, pl)
            } else {
                // Online construction
                into_keys = into_keys.rewind()?;
                into_values = into_values.rewind()?;
                let mut sig_vals = Vec::with_capacity(self.expected_num_keys.unwrap_or(0));
                while let Some(result) = into_keys.next() {
                    match result {
                        Ok(key) => {
                            pl.light_update();
                            let v = into_values.next().expect("Not enough values")?;
                            max_value = Ord::max(max_value, *v);
                            sig_vals.push(SigVal {
                                sig: T::to_sig(key, seed),
                                val: *v,
                            });
                        }
                        Err(e) => {
                            bail!("Error reading input: {}", e);
                        }
                    }
                }
                pl.done();
                self.num_keys = sig_vals.len();
                self.bit_width = max_value.len() as usize;

                self.set_up_shards();

                pl.start("Sorting...");
                sig_vals.par_sort_unstable_by(|a, b| {
                    a.sig
                        .shard(self.shard_high_bits, self.shard_mask)
                        .cmp(&b.sig.shard(self.shard_high_bits, self.shard_mask))
                });
                pl.done_with_count(self.num_keys);

                pl.start("Counting shard sizes...");
                let mut shard_sizes = vec![0; self.num_shards];
                for &sig_val in &sig_vals {
                    shard_sizes[sig_val.sig.shard(self.shard_high_bits, self.shard_mask)] += 1;
                }
                pl.done_with_count(self.num_keys);

                let max_shard = shard_sizes.iter().copied().max().unwrap_or(0);

                pl.info(format_args!(
                    "Keys: {} Max value: {} Bit width: {} Shards: 2^{} Max shard / average shard: {:.2}%",
                    self.num_keys, max_value, self.bit_width, self.shard_high_bits,
                    (100.0 * max_shard as f64) / (self.num_keys as f64 / self.num_shards as f64)
                ));

                self.build_iter(
                    seed,
                    IntoChunkIter {
                        data: sig_vals,
                        shard_sizes,
                    },
                    max_shard,
                    new,
                    pl,
                )
            } {
                Ok(result) => {
                    pl.info(format_args!(
                        "Completed in {:.3} seconds",
                        start.elapsed().as_secs_f64()
                    ));
                    return Ok(result);
                }
                // Let's try another seed, but just a few times--most likely,
                // duplicate keys
                Err(SolveError::DuplicateSignature) => {
                    if dup_count >= 3 {
                        bail!("Duplicate keys (duplicate 128-bit signatures with four different seeds)");
                    }
                    pl.warn(format_args!(
                        "Duplicate 128-bit signature, trying again with a different seed..."
                    ));
                    dup_count += 1;
                    continue;
                }
                Err(SolveError::MaxShardTooBig) => {
                    pl.warn(format_args!(
                        "The maximum shard is too big, trying again with a different seed..."
                    ));
                }
                // Let's just try another seed
                Err(SolveError::UnsolvableShard) => {
                    pl.warn(format_args!(
                        "Unsolvable shard, trying again with different seed..."
                    ));
                }
            };

            seed += 1;
        }
    }

    /// Build and return a new function starting from an iterator on shards.
    ///
    /// This method provide construction logic that is independent from the
    /// actual storage of the values (offline or in core memory.)
    ///
    /// See [`VFuncBuilder::_build`] for more details on the parameters.
    fn build_iter<'a, I, B, P>(
        &mut self,
        seed: u64,
        mut into_shard_iter: I,
        max_shard: usize,
        new: fn(usize, usize) -> D,
        pl: &mut P,
    ) -> Result<VFunc<T, W, D, S, SHARDED>, SolveError>
    where
        P: ProgressLog + Clone + Send + Sync,
        B: BorrowMut<[SigVal<W>]> + Send,
        I: 'a,
        &'a mut I: IntoIterator<Item = B>,
        <&'a mut I as IntoIterator>::IntoIter: Send,
    {
        let (lazy_gaussian, c);
        if SHARDED {
            lazy_gaussian = self.num_keys <= MAX_LIN_SIZE;

            (c, self.log2_seg_size) = if lazy_gaussian {
                // Slightly loose bound to help with solvability
                (1.11, lin_log2_seg_size(3, max_shard))
            } else {
                (1.10, fuse_log2_seg_size(3, max_shard))
            };
        } else {
            lazy_gaussian = self.num_keys <= MAX_LIN_SHARD_SIZE;
            c = fuse_c(3, self.num_keys);
            self.log2_seg_size = if lazy_gaussian {
                lin_log2_seg_size(3, self.num_keys)
            } else {
                fuse_log2_seg_size(3, self.num_keys)
            };
        }

        self.l = ((c * max_shard as f64).ceil() as usize).div_ceil(1 << self.log2_seg_size);
        self.num_vertices = (1 << self.log2_seg_size) * (self.l + 2);

        let mut data = new(self.bit_width, self.num_vertices * self.num_shards);

        let thread_pool = ThreadPoolBuilder::new()
            .num_threads(self.num_shards.min(self.max_num_threads) + 1) // Or it might hang
            .build()
            .unwrap(); // Seroiusly, it's not going to fail

        pl.info(format_args!(
            "c: {}, log₂ segment size: {} Number of variables: {:.2}% Number of threads: {}",
            c,
            self.log2_seg_size,
            (100.0 * (self.num_vertices * self.num_shards) as f64) / (self.num_keys as f64),
            thread_pool.current_num_threads()
        ));

        // Loop until success or duplicate detection
        // This is safe because the reference disappears at the end of the
        // body of the loop
        let iter = unsafe { (*(&mut into_shard_iter as *mut I)).into_iter().enumerate() };

        if lazy_gaussian {
            pl.info(format_args!("Switching to lazy Gaussian elimination"));

            self.par_solve(
                iter,
                &mut data,
                new,
                |this, shard_index, shard, data, pl| this.solve_lin(shard_index, shard, data, pl),
                &thread_pool,
                &mut pl.concurrent(),
                pl,
            )?;
        } else {
            self.par_solve(
                iter,
                &mut data,
                new,
                |this, shard_index, shard, data, pl| {
                    this.peel_shard(shard_index, shard, data, pl)
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

        Ok(VFunc::<T, W, D, S, SHARDED> {
            seed,
            l: self.l,
            shard_high_bits: self.shard_high_bits,
            shard_mask: self.shard_mask,
            num_keys: self.num_keys,
            log2_seg_size: self.log2_seg_size,
            data,
            _marker_t: std::marker::PhantomData,
            _marker_os: std::marker::PhantomData,
        })
    }
}
#[cfg(test)]
mod tests {
    use rand::{rngs::SmallRng, Rng, SeedableRng};

    #[test]
    #[cfg(feature = "slow_tests")]
    fn test_lin_log2_seg_size() -> anyhow::Result<()> {
        use super::*;
        // Manually tested values:
        // 200000 -> 9
        // 150000 -> 9
        // 112500 -> 9
        // 100000 -> 8
        // 75000 -> 8
        // 50000 -> 8
        // 40000 -> 8
        // 35000 -> 7
        // 30000 -> 7
        // 25000 -> 7
        // 12500 -> 7
        // 10000 -> 6
        // 1000 -> 5
        // 100 -> 3
        // 10 -> 2

        for n in 0..MAX_LIN_SHARD_SIZE * 2 {
            if lin_log2_seg_size(3, n) != lin_log2_seg_size(3, n + 1) {
                eprintln!(
                    "Bulding function with {} keys (log₂ segment size = {})...",
                    n,
                    lin_log2_seg_size(3, n)
                );
                let _func = VFuncBuilder::<_, _, BitFieldVec<_>, [u64; 2], true>::default().build(
                    FromIntoIterator::from(0..n),
                    FromIntoIterator::from(0_usize..),
                    no_logging![],
                )?;

                eprintln!(
                    "Bulding function with {} keys (log₂ segment size = {})...",
                    n + 1,
                    lin_log2_seg_size(3, n + 1)
                );
                let _func = VFuncBuilder::<_, _, BitFieldVec<_>, [u64; 2], true>::default().build(
                    FromIntoIterator::from(0..n + 1),
                    FromIntoIterator::from(0_usize..),
                    no_logging![],
                )?;
            }
        }

        Ok(())
    }

    #[test]
    fn test_count_sort() {
        use super::*;
        let mut rng = SmallRng::seed_from_u64(0);

        for high_bits in [1, 8, 16] {
            eprintln!("Testing with {} high bits...", high_bits);
            let mut data = (0..100000).map(|_| rng.random()).collect::<Vec<u64>>();
            let counts = count_sort(&mut data, 1 << high_bits, |x| {
                (x >> (u64::BITS - high_bits)) as usize
            });

            let mut my_counts = vec![0; 1 << high_bits];
            for &x in &data {
                my_counts[(x >> (u64::BITS - high_bits)) as usize] += 1;
            }

            assert_eq!(counts, my_counts);
            for i in 1..data.len() {
                assert!(
                    data[i - 1] >> (u64::BITS - high_bits) <= data[i] >> (u64::BITS - high_bits)
                );
            }
        }
    }
}
