/*
*
* SPDX-FileCopyrightText: 2023 Sebastiano Vigna
*
* SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
*/

// We import selectively to be able to use AtomicHelper
use crate::bits::*;
use crate::traits::bit_field_slice;
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
use log::warn;
use mem_dbg::*;
use mucow::MuCow;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use rdst::*;
use std::marker::PhantomData;

#[inline]
pub fn comp_log2_seg_size(arity: usize, num_keys: usize) -> u32 {
    if num_keys == 0 {
        return 2;
    }

    match arity {
        3 => ((num_keys as f64).ln() / (3.33_f64).ln() + 2.25).floor() as u32,
        4 => ((num_keys as f64).ln() / (2.91_f64).ln() - 0.5).floor() as u32,
        _ => unimplemented!(),
    }
}

#[inline]
pub fn size_factor(arity: usize, size: usize) -> f64 {
    match arity {
        3 => 1.125_f64.max(0.875 + 0.25 * (1000000_f64).ln() / (size as f64).ln()),
        4 => 1.075_f64.max(0.77 + 0.305 * (600000_f64).ln() / (size as f64).ln()),
        _ => unimplemented!(),
    }
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
        debug_assert!(self.degree() == 0);
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
    high_bits: u32,
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
    #[setters(generate = true)]
    /// The number of parallel threads to use.
    num_threads: usize,
    #[setters(generate = true)]
    /// Use disk-based buckets to reduce core memory usage at construction time.
    offline: bool,
    /// The base-2 logarithm of the number of buckets. Used only if `offline` is
    /// `true`. The default is 8.
    #[setters(generate = true, strip_option)]
    log2_buckets: Option<u32>,
    bit_width: usize,
    shard_high_bits: u32,
    shard_mask: u32,
    num_keys: usize,
    num_shards: usize,
    num_vertices: usize,
    log2_seg_size: u32,
    l: usize,

    _marker_t: PhantomData<T>,
    _marker_od: PhantomData<(W, D, S)>,
}

/// Given the number of keys, and whether sharding should be used, returns the
/// number of high bits to use for sharding.
///
/// The shard size is computed so that the largest shard is with high probabiliy
/// at most 1.001% the size of the average shard.
fn compute_high_bits(num_keys: usize, sharded: bool) -> u32 {
    let eps = 0.001;
    if sharded {
        {
            let t = (num_keys as f64 * eps * eps / 2.0).ln();

            if t > 0.0 {
                ((t - t.ln()) / 2_f64.ln()).ceil()
            } else {
                0.0
            }
        }
        .min(10.0) as u32 // More than 1000 shards make no sense}
    } else {
        0
    }
}

#[derive(thiserror::Error, Debug)]
/// Errors that can happen during deserialization.
pub enum SolveError {
    #[error("Duplicate signature")]
    /// A duplicate signature was detected.
    DuplicateSignature,
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
        I: Iterator<Item = (usize, MuCow<'a, [SigVal<W>]>)> + Send,
        C: ConcurrentProgressLog + Send + Sync,
        P: ProgressLog + Clone + Send + Sync,
    >(
        &mut self,
        main_pl: &mut C,
        pl: &mut P,
        shard_iter: I,
        data: &mut D,
        new_data: impl Fn(usize, usize) -> D + Sync,
        solve_shard: impl Fn(&Self, usize, MuCow<'a, [SigVal<W>]>, D, &mut P) -> Result<(usize, D), ()>
            + Send
            + Sync,
        thread_pool: &rayon::ThreadPool,
    ) -> Result<(), SolveError>
    where
        SigVal<W>: RadixKey + Send + Sync,
    {
        // TODO: optimize for the non-parallel case
        let (send, receive) = crossbeam_channel::bounded::<(usize, D)>(2 * self.num_threads);

        thread_pool.scope(|scope| {
            scope.spawn(|_| {
                for (shard_index, shard_data) in receive {
                    // TODO: fast copy
                    for i in 0..self.num_vertices {
                        data.set(shard_index * self.num_vertices + i, shard_data.get(i));
                    }
                }
            });

            shard_iter
                .par_bridge()
                .try_for_each_with(
                    (main_pl.clone(), pl.clone()),
                    |(main_pl, pl), (shard_index, mut shard)| {
                        main_pl.info(format_args!(
                            "Sorting and checking shard {}/{}",
                            shard_index + 1,
                            self.num_shards
                        ));

                        shard.radix_sort_unstable();

                        if shard.par_windows(2).any(|w| w[0].sig == w[1].sig) {
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
        })
    }

    /// Solve a shard by peeling.
    ///
    /// Return the shard index and the data, in case of success,
    /// or the shard index, the shard, the edge lists, the data, and the stack
    /// at the end of the peeling procedure in case of failure. These data can
    /// be then passed to a linear solver to complete the solution.
    fn peel_shard<'a>(
        &self,
        shard_index: usize,
        shard: MuCow<'a, [SigVal<W>]>,
        data: D,
        pl: &mut impl ProgressLog,
    ) -> Result<(usize, D), (usize, MuCow<'a, [SigVal<W>]>, Vec<EdgeList>, D, Vec<usize>)> {
        pl.start(format!(
            "Generating graph for shard {}/{}...",
            shard_index + 1,
            self.num_shards
        ));
        let mut edge_lists = Vec::new();
        edge_lists.resize_with(self.num_vertices, EdgeList::default);
        shard.iter().enumerate().for_each(|(edge_index, sig_val)| {
            for (side, &v) in sig_val
                .sig
                .edge(self.shard_high_bits, self.l, self.log2_seg_size)
                .iter()
                .enumerate()
            {
                edge_lists[v].add(edge_index, side);
            }
        });
        pl.done_with_count(shard.len());

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

                let e =
                    shard[edge_index]
                        .sig
                        .edge(self.shard_high_bits, self.l, self.log2_seg_size);
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
            warn!(
                "Peeling failed for shared {}/{} (peeled {} out of {} edges)",
                shard_index + 1,
                self.num_shards,
                stack.len(),
                shard.len()
            );
            return Err((shard_index, shard, edge_lists, data, stack));
        }
        pl.done_with_count(shard.len());

        Ok(self.assign(shard_index, shard, data, edge_lists, stack, pl))
    }

    /// Perform assignment of values based on peeling data.
    ///
    /// This method might be called after a successful peeling procedure,
    /// or after a linear solver has been used to solve the remaining edges.
    fn assign(
        &self,
        shard_index: usize,
        shard: MuCow<[SigVal<W>]>,
        mut data: D,
        edge_lists: Vec<EdgeList>,
        mut stack: Vec<usize>,
        pl: &mut impl ProgressLog,
    ) -> (usize, D) {
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
        shard: MuCow<'a, [SigVal<W>]>,
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
                let mut peeled = vec![false; shard.len()];
                stack
                    .iter()
                    .filter(|&v| edge_lists[*v].degree() == 0)
                    .for_each(|&v| {
                        peeled[edge_lists[v].edge_index_and_hinge().0] = true;
                    });

                // Likely--we have solve the rest
                pl.start(format!(
                    "Generating system for shard {}/{}...",
                    shard_index + 1,
                    self.num_shards
                ));
                // Create F₂ system using non-peeled edges
                let mut system = Modulo2System::<W>::new(self.num_vertices);
                shard
                    .iter()
                    .enumerate()
                    .filter(|(edge_index, _)| !peeled[*edge_index])
                    .for_each(|(_, sig_val)| {
                        let mut eq = Modulo2Equation::<W>::new(sig_val.val, self.num_vertices);
                        for &v in sig_val
                            .sig
                            .edge(self.shard_high_bits, self.l, self.log2_seg_size)
                            .iter()
                        {
                            eq.add(v);
                        }
                        system.add(eq);
                    });
                pl.done_with_count(shard.len());

                pl.start("Solving system...");
                let result = system
                    .lazy_gaussian_elimination_constructor()
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
            let edge = sig.edge(self.high_bits, self.l, self.log2_seg_size);
            let chunk = sig.shard(self.high_bits, self.shard_mask);
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
    type IntoIter = std::iter::Map<ArbitraryChunkMut<'a, 'a, T>, fn(&mut [T]) -> MuCow<'_, [T]>>;
    type Item = MuCow<'a, [T]>;

    fn into_iter(self) -> Self::IntoIter {
        self.data
            .arbitrary_chunks_mut(&self.shard_sizes)
            .map(|s| MuCow::Borrowed(s))
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
        let mut _dup_count = 0; // TODO
        let mut seed = 0;
        // Loop until success or duplicate detection
        loop {
            pl.item_name("key");
            pl.start("Reading input...");
            let mut max_value = W::ZERO;

            if self.offline {
                // Offline construction using a SigStore
                let max_shard_high_bits = 12;
                let log2_buckets = self.log2_buckets.unwrap_or(8);
                pl.info(format_args!("Using {} buckets", 1 << log2_buckets));
                // Put values into a signature sorter
                let mut sig_sorter = SigStore::<W>::new(log2_buckets, max_shard_high_bits).unwrap();
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
                self.shard_high_bits = compute_high_bits(self.num_keys, SHARDED);
                let shard_store = sig_sorter.into_chunk_store(self.shard_high_bits)?;
                let max_chunk = *shard_store.chunk_sizes().iter().max().unwrap();
                self.build_iter(seed, shard_store, max_chunk, new, pl);

                /*Ok(()) => break data,
                    Err(_) => {
                        if dup_count >= 3 {
                            bail!("Duplicate keys (duplicate 128-bit signatures with four different seeds");
                        }
                        warn!("Duplicate 128-bit signature, trying again...");
                        dup_count += 1;
                        continue;
                    }
                }*/
            } else {
                // Online construction
                into_keys = into_keys.rewind()?;
                into_values = into_values.rewind()?;
                let mut sig_vals = vec![];
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
                let lazy_gaussian = self.num_keys < 60000000;

                self.shard_high_bits = if lazy_gaussian {
                    // log(num_keys / 65536) + 1
                    self.num_keys.ilog2().saturating_sub(15)
                } else {
                    compute_high_bits(self.num_keys, SHARDED)
                };

                dbg!(self.shard_high_bits);
                self.num_threads = 1.max((1 << self.shard_high_bits) / 8).min(8); // TODO
                                                                                  //(self.shard_high_bits, self.num_threads) = (0, 4);
                self.log2_seg_size = comp_log2_seg_size(3, self.num_keys >> self.shard_high_bits);
                dbg!(self.log2_seg_size);
                let c = 1.10; // size_factor(3, num_keys);
                dbg!(c, self.log2_seg_size);

                self.num_shards = 1 << self.shard_high_bits;
                self.shard_mask = (1u32 << self.shard_high_bits) - 1;

                // TODO: this might be a countsort on the shard_high_bits
                pl.start("Sorting...");
                sig_vals.radix_sort_unstable();
                pl.done_with_count(self.num_keys);

                pl.start("Counting chunk sizes...");

                let mut shard_sizes = vec![0_usize; self.num_shards];

                for &w in &sig_vals {
                    shard_sizes[w.sig.shard(self.shard_high_bits, self.shard_mask)] += 1;
                }

                pl.done_with_count(self.num_keys);

                /*if dup {
                    if dup_count >= 3 {
                        bail!("Duplicate keys (duplicate 128-bit signatures with four different seeds)");
                    }
                    warn!("Duplicate 128-bit signature, trying again...");
                    dup_count += 1;
                    continue;
                }*/

                self.bit_width = max_value.len() as usize;
                pl.info(format_args!(
                    "max value = {}, bit width = {}",
                    max_value, self.bit_width
                ));

                /*pl.info(format_args!(
                    "Number of variables with respect to c*n: {:.2}%",
                    (100.0 * (self.num_vertices * self.num_chunks) as f64)
                        / (self.num_keys as f64 * c)
                ));*/

                let max_chunk = *shard_sizes.iter().max().unwrap();

                self.build_iter(
                    seed,
                    IntoChunkIter {
                        data: sig_vals,
                        shard_sizes,
                    },
                    max_chunk,
                    new,
                    pl,
                );
            }
            seed += 1;
        }
    }

    /// Build and return a new function starting from an iterator on shards.
    ///
    /// This method provide construction logic that is independent from the
    /// actual storage of the values (offline or in core memory.)
    ///
    /// See [`VFuncBuilder::_build`] for more details on the parameters.
    fn build_iter<'a, 'b, I>(
        &mut self,
        seed: u64,
        mut into_shard_iter: I,
        max_shard: usize,
        new: fn(usize, usize) -> D,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> anyhow::Result<VFunc<T, W, D, S, SHARDED>>
    where
        I: 'static,
        &'a mut I: IntoIterator<Item = MuCow<'a, [SigVal<W>]>>,
        <&'a mut I as IntoIterator>::IntoIter: Send,
    {
        let mut _dup_count = 0;

        let c = 1.10; // size_factor(3, num_keys);
        self.l = ((c * max_shard as f64).ceil() as usize).div_ceil(1 << self.log2_seg_size);

        let mut data = new(self.bit_width, self.num_vertices * self.num_shards);

        let thread_pool = ThreadPoolBuilder::new()
            .num_threads(self.num_threads.max(2)) // Or it might hang
            .build()?;

        // Loop until success or duplicate detection
        let data = loop {
            let lazy_gaussian = self.num_keys < 60000000;

            self.shard_high_bits = if lazy_gaussian {
                // log(num_keys / 65536) + 1
                self.num_keys.ilog2().saturating_sub(15)
            } else {
                compute_high_bits(self.num_keys, SHARDED)
            };
            dbg!(self.shard_high_bits);
            self.num_threads = 1.max((1 << self.shard_high_bits) / 8).min(8); // TODO
                                                                              //(self.shard_high_bits, self.num_threads) = (0, 4);
            self.log2_seg_size = comp_log2_seg_size(3, self.num_keys >> self.shard_high_bits);
            dbg!(self.log2_seg_size);
            dbg!(c, self.log2_seg_size);

            self.num_shards = 1 << self.shard_high_bits;
            self.shard_mask = (1u32 << self.shard_high_bits) - 1;

            dbg!(self.l);
            self.num_vertices = (1 << self.log2_seg_size) * (self.l + 2);
            dbg!(self.num_vertices);

            pl.info(format_args!(
                "Number of variables with respect to c*n: {:.2}%",
                (100.0 * (self.num_vertices * self.num_shards) as f64) / (self.num_keys as f64 * c)
            ));

            let iter = into_shard_iter.into_iter().enumerate();
            if lazy_gaussian {
                self.l = self.l.saturating_sub(1);
                pl.info(format_args!("Switching to lazy Gaussian elmination"));

                match self.par_solve(
                    &mut pl.concurrent(),
                    pl,
                    iter,
                    &mut data,
                    new,
                    |this, shard_index, shard, data, pl| {
                        this.solve_lin(shard_index, shard, data, pl)
                    },
                    &thread_pool,
                ) {
                    Ok(()) => break data,
                    Err(_) => continue,
                }
            } else {
                match self.par_solve(
                    &mut pl.concurrent(),
                    pl,
                    iter,
                    &mut data,
                    new,
                    |this, shard_index, shard, data, pl| {
                        this.peel_shard(shard_index, shard, data, pl)
                            .map_err(|_| ())
                    },
                    &thread_pool,
                ) {
                    Ok(()) => break data,
                    Err(_) => {}
                }
            }
        };
        pl.info(format_args!(
            "bits/keys: {} ({:.2}%)",
            data.len() as f64 * self.bit_width as f64 / self.num_keys as f64,
            data.len() as f64 / (self.num_keys as f64 * 1.10)
        ));

        Ok(VFunc::<T, W, D, S, SHARDED> {
            seed,
            l: self.l,
            high_bits: self.shard_high_bits,
            shard_mask: self.shard_mask,
            num_keys: self.num_keys,
            log2_seg_size: self.log2_seg_size,
            data,
            _marker_t: std::marker::PhantomData,
            _marker_os: std::marker::PhantomData,
        })
    }
}
