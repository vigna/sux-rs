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
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use rdst::*;
use std::borrow::Cow;
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

    #[inline(always)]
    fn degree(&self) -> usize {
        (self.0 >> Self::DEG_SHIFT) as usize
    }

    #[inline(always)]
    fn edge_index_side(&self) -> (usize, usize) {
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

- the `high_bits` most significant bits of `h` are used to select a chunk;
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
    O: ZeroCopy + Word = usize,
    D: bit_field_slice::BitFieldSlice<O> = BitFieldVec<O>,
    S = [u64; 2],
    const SHARDED: bool = false,
> {
    high_bits: u32,
    log2_seg_size: u32,
    chunk_mask: u32,
    seed: u64,
    l: usize,
    num_keys: usize,
    data: D,
    _marker_t: std::marker::PhantomData<T>,
    _marker_os: std::marker::PhantomData<(O, S)>,
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
/// chunk of the signatures; for each signature in a chunk, a thread will
/// allocate about two `usize` values in core memory; in the case of offline
/// construction, also signatures and values in the chunk will be stored in core
/// memory.
///
/// For very large key sets chunks will be significantly smaller than the number
/// of keys, so the memory usage, in particular in offline mode, can be
/// significantly reduced. Note that using too many threads might actually be
/// harmful due to memory contention: eight is usually a good value.
#[derive(Setters, Debug, Derivative)]
#[derivative(Default)]
#[setters(generate = false)]
pub struct VFuncBuilder<
    T: ?Sized + Send + Sync + ToSig,
    O: ZeroCopy + Word,
    D: bit_field_slice::BitFieldSlice<O> + Send + Sync = BitFieldVec<O>,
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
    chunk_high_bits: u32,
    chunk_mask: u32,
    num_keys: usize,
    num_chunks: usize,
    num_vertices: usize,
    log2_seg_size: u32,
    l: usize,

    _marker_t: PhantomData<T>,
    _marker_od: PhantomData<(O, D, S)>,
}

fn compute_params(num_keys: usize, sharded: bool) -> (u32, usize) {
    let eps = 0.001;
    let chunk_high_bits = if sharded {
        {
            let t = (num_keys as f64 * eps * eps / 2.0).ln();

            if t > 0.0 {
                ((t - t.ln()) / 2_f64.ln()).ceil()
            } else {
                0.0
            }
        }
        .min(10.0) as u32 // More than 1000 chunks make no sense}
    } else {
        0
    };
    let max_num_threads = 1.max((1 << chunk_high_bits) / 8).min(8); // TODO

    dbg!((chunk_high_bits, max_num_threads))
}

impl<
        T: ?Sized + Send + Sync + ToSig,
        O: ZeroCopy + Word + Send + Sync,
        D: BitFieldSlice<O> + BitFieldSliceMut<O> + Send + Sync,
        S: Send + Sync,
        const SHARDED: bool,
    > VFuncBuilder<T, O, D, S, SHARDED>
{
    #[allow(clippy::too_many_arguments)]
    fn par_solve<
        'a,
        I: Iterator<Item = (usize, Cow<'a, [SigVal<O>]>)> + Send,
        C: ConcurrentProgressLog + Send + Sync,
        P: ProgressLog + Clone + Send + Sync,
    >(
        &mut self,
        main_pl: &mut C,
        pl: &mut P,
        chunk_iter: I,
        data: &mut D,
        new: impl Fn(usize, usize) -> D + Sync,
        do_chunk: impl Fn(
                &Self,
                &mut C,
                &mut P,
                usize,
                Cow<'a, [SigVal<O>]>,
                D,
            ) -> Result<(usize, D), (usize, D, Vec<usize>)>
            + Send
            + Sync,
    ) -> anyhow::Result<()>
    where
        SigVal<O>: RadixKey + Send + Sync,
    {
        self.chunk_high_bits = self.num_chunks.ilog2();
        let (send, receive) = crossbeam_channel::bounded::<(usize, D)>(2 * self.num_threads);

        ThreadPoolBuilder::new()
            .num_threads(self.num_threads)
            .build()?
            .scope(|scope| {
                scope.spawn(|_| {
                    for (chunk_index, chunk_data) in receive {
                        for i in 0..self.num_vertices {
                            data.set(chunk_index * self.num_vertices + i, chunk_data.get(i));
                        }
                    }
                });

                chunk_iter
                    .par_bridge()
                    .try_for_each_with(
                        (main_pl.clone(), pl.clone()),
                        |(main_pl, pl), (chunk_index, chunk)| {
                            do_chunk(
                                self,
                                main_pl,
                                pl,
                                chunk_index,
                                chunk,
                                new(self.bit_width, self.num_vertices),
                            )
                            .map(|(chunk_index, data)| {
                                send.send((chunk_index, data)).unwrap();
                            })
                        },
                    )
                    .map(|_| {
                        drop(send);
                    })
            })
            .map_err(|_e| anyhow::anyhow!("Couldn't solve"))?; // TODO
        Ok(())
    }

    fn peel_chunk<'a>(
        &self,
        main_pl: &mut impl ProgressLog,
        pl: &mut impl ProgressLog,
        chunk_index: usize,
        mut chunk: Cow<'a, [SigVal<O>]>,
        data: D,
    ) -> Result<(usize, D), (usize, D, Vec<usize>)> {
        if let Cow::Owned(chunk) = &mut chunk {
            chunk.radix_sort_unstable();
        }

        /*if chunk.par_windows(2).any(|w| w[0].sig == w[1].sig) { TODO
            bail!("Duplicate signature in chunk");
        }*/

        pl.start(format!(
            "Generating graph for chunk {}/{}...",
            chunk_index + 1,
            self.num_chunks
        ));
        let mut edge_lists = Vec::new();
        edge_lists.resize_with(self.num_vertices, EdgeList::default);
        chunk.iter().enumerate().for_each(|(edge_index, sig_val)| {
            for (side, &v) in sig_val
                .sig
                .edge(self.chunk_high_bits, self.l, self.log2_seg_size)
                .iter()
                .enumerate()
            {
                edge_lists[v].add(edge_index, side);
            }
        });
        pl.done_with_count(chunk.len());

        pl.start(format!(
            "Peeling graph for chunk {}/{}...",
            chunk_index + 1,
            self.num_chunks
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
                let (edge_index, side) = edge_lists[v].edge_index_side();
                edge_lists[v].zero();
                stack[curr] = v;
                curr += 1;

                let e =
                    chunk[edge_index]
                        .sig
                        .edge(self.chunk_high_bits, self.l, self.log2_seg_size);
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
        if chunk.len() != stack.len() {
            /*warn!(
                "Peeling failed for chunk {}/{} (peeled {} out of {} edge)",
                chunk_index + 1,
                num_chunks,
                stack.len(),
                chunk.len()
            );*/
            return Err((chunk_index, data, stack));
        }
        pl.done_with_count(chunk.len());

        let assignment = self.assign(chunk_index, chunk, data, edge_lists, stack, pl);
        main_pl.info(format_args!(
            "Completed chunk {}/{}.",
            chunk_index + 1,
            self.num_chunks
        ));
        main_pl.update_and_display();
        Ok(assignment)
    }

    fn assign(
        &self,
        chunk_index: usize,
        chunk: Cow<[SigVal<O>]>,
        mut data: D,
        edge_lists: Vec<EdgeList>,
        mut stack: Vec<usize>,
        pl: &mut impl ProgressLog,
    ) -> (usize, D) {
        pl.start(format!(
            "Assigning values for chunk {}/{}...",
            chunk_index + 1,
            self.num_chunks
        ));
        while let Some(v) = stack.pop() {
            let (edge_index, side) = edge_lists[v].edge_index_side();
            let edge = chunk[edge_index]
                .sig
                .edge(self.chunk_high_bits, self.l, self.log2_seg_size);
            let value = match side {
                0 => data.get(edge[1]) ^ data.get(edge[2]),
                1 => data.get(edge[0]) ^ data.get(edge[2]),
                2 => data.get(edge[0]) ^ data.get(edge[1]),
                _ => unreachable!(),
            };

            data.set(v, chunk[edge_index].val ^ value);
            debug_assert_eq!(
                data.get(edge[0]) ^ data.get(edge[1]) ^ data.get(edge[2]),
                chunk[edge_index].val
            );
        }
        pl.done_with_count(chunk.len());

        (chunk_index, data)
    }
}

pub trait ChunkEdge: Send + Sync {
    fn chunk(&self, chunk_high_bits: u32, chunk_mask: u32) -> usize;
    fn edge(&self, chunk_high_bits: u32, l: usize, log2_seg_size: u32) -> [usize; 3];
}

impl ChunkEdge for [u64; 2] {
    #[inline(always)]
    #[must_use]
    fn chunk(&self, chunk_high_bits: u32, chunk_mask: u32) -> usize {
        (self[0].rotate_left(chunk_high_bits) & chunk_mask as u64) as usize
    }

    #[inline(always)]
    #[must_use]
    fn edge(&self, chunk_high_bits: u32, l: usize, log2_seg_size: u32) -> [usize; 3] {
        let first_segment = ((self[0] << chunk_high_bits >> 32) * l as u64 >> 32) as usize;
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
    fn chunk(&self, chunk_high_bits: u32, chunk_mask: u32) -> usize {
        let xor = *self as u32 ^ (*self >> 32) as u32;
        (xor.rotate_left(chunk_high_bits) & chunk_mask) as usize
    }

    #[inline(always)]
    #[must_use]
    fn edge(&self, chunk_high_bits: u32, l: usize, log2_seg_size: u32) -> [usize; 3] {
        let xor = *self as u32 ^ (*self >> 32) as u32;
        let first_segment = ((xor.rotate_left(chunk_high_bits) as u64 * l as u64) >> 32) as usize;
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
        O: ZeroCopy + Word,
        D: bit_field_slice::BitFieldSlice<O>,
        S: ChunkEdge,
        const SHARDED: bool,
    > VFunc<T, O, D, S, SHARDED>
{
    /// Return the value associated with the given signature.
    ///
    /// This method is mainly useful in the construction of compound functions.
    #[inline(always)]
    pub fn get_by_sig(&self, sig: &[u64; 2]) -> O {
        if SHARDED {
            let edge = sig.edge(self.high_bits, self.l, self.log2_seg_size);
            let chunk = sig.chunk(self.high_bits, self.chunk_mask);
            // chunk * self.segment_size * (2^log2_l + 2)
            let chunk_offset = chunk * ((self.l + 2) << self.log2_seg_size);

            unsafe {
                self.data.get_unchecked(edge[0] + chunk_offset)
                    ^ self.data.get_unchecked(edge[1] + chunk_offset)
                    ^ self.data.get_unchecked(edge[2] + chunk_offset)
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
    pub fn get(&self, key: &T) -> O {
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

impl<T: ?Sized + Send + Sync + ToSig, O: ZeroCopy + Word, S: ChunkEdge, const SHARDED: bool>
    VFuncBuilder<T, O, Vec<O>, S, SHARDED>
where
    SigVal<O>: RadixKey + Send + Sync,
    Vec<O>: BitFieldSliceMut<O> + BitFieldSlice<O>,
{
    pub fn build(
        self,
        into_keys: impl RewindableIoLender<T>,
        into_values: impl RewindableIoLender<O>,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> anyhow::Result<VFunc<T, O, Vec<O>, S>> {
        self._build(
            into_keys,
            into_values,
            |_bit_width: usize, len: usize| vec![O::ZERO; len],
            pl,
        )
    }
}

impl<T: ?Sized + Send + Sync + ToSig, O: ZeroCopy + Word, S: ChunkEdge, const SHARDED: bool>
    VFuncBuilder<T, O, BitFieldVec<O>, S, SHARDED>
where
    SigVal<O>: RadixKey + Send + Sync,
{
    pub fn build(
        self,
        into_keys: impl RewindableIoLender<T>,
        into_values: impl RewindableIoLender<O>,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> anyhow::Result<VFunc<T, O, BitFieldVec<O>, S>> {
        self._build(
            into_keys,
            into_values,
            |bit_width: usize, len: usize| BitFieldVec::<O>::new(bit_width, len),
            pl,
        )
    }
}

impl<
        T: ?Sized + Send + Sync + ToSig,
        O: ZeroCopy + Word,
        D: bit_field_slice::BitFieldSlice<O> + bit_field_slice::BitFieldSliceMut<O> + Send + Sync,
        S: ChunkEdge,
        const SHARDED: bool,
    > VFuncBuilder<T, O, D, S, SHARDED>
where
    SigVal<O>: RadixKey + Send + Sync,
{
    /// Build and return a new function with given keys and values.
    fn _build(
        mut self,
        mut into_keys: impl RewindableIoLender<T>,
        mut into_values: impl RewindableIoLender<O>,
        new: fn(usize, usize) -> D,
        pl: &mut (impl ProgressLog + Clone + Send + Sync),
    ) -> anyhow::Result<VFunc<T, O, D, S>> {
        // Loop until success or duplicate detection
        let mut dup_count = 0;
        let mut seed = 0;
        let data = loop {
            pl.item_name("key");
            pl.start("Reading input...");
            let mut max_value = O::ZERO;
            //let mut chunk_sizes;
            let (max_num_threads, c);

            //let (input_time, build_time);

            if self.offline {
                let max_chunk_high_bits = 12;
                let log2_buckets = self.log2_buckets.unwrap_or(8);
                pl.info(format_args!("Using {} buckets", 1 << log2_buckets));
                let mut sig_sorter = SigStore::<O>::new(log2_buckets, max_chunk_high_bits).unwrap();
                into_values = into_values.rewind()?;
                into_keys = into_keys.rewind()?;
                while let Some(result) = into_keys.next() {
                    match result {
                        Ok(key) => {
                            pl.light_update();
                            let v = into_values.next().expect("Not enough values")?;
                            max_value = Ord::max(max_value, *v);
                            sig_sorter.push(&SigVal {
                                sig: T::to_sig(key, seed),
                                val: *v,
                            })?;
                        }
                        Err(e) => {
                            bail!("Error reading input: {}", e);
                        }
                    }
                }
                self.num_keys = sig_sorter.len();
                pl.done();

                (self.chunk_high_bits, max_num_threads) = compute_params(self.num_keys, SHARDED);
                //chunk_high_bits = 0;
                //max_num_threads = 1;
                self.log2_seg_size = comp_log2_seg_size(3, self.num_keys);
                c = 1.10; // TODO size_factor(3, num_keys);

                dbg!(c, self.log2_seg_size);

                self.num_chunks = 1 << self.chunk_high_bits;
                self.chunk_mask = (1u32 << self.chunk_high_bits) - 1;

                let mut chunk_store = sig_sorter.into_chunk_store(self.chunk_high_bits)?;
                // let chunk_sizes = chunk_store.chunk_sizes();

                self.bit_width = max_value.len() as usize;
                pl.info(format_args!(
                    "max value = {}, bit width = {}",
                    max_value, self.bit_width
                ));

                self.l =
                    ((self.num_keys as f64 * c).ceil() as usize).div_ceil(1 << self.log2_seg_size);
                self.num_vertices = (1 << self.log2_seg_size) * (self.l + 2);
                pl.info(format_args!(
                    "Size {:.2}%",
                    (100.0 * (self.num_vertices * self.num_chunks) as f64)
                        / (self.num_keys as f64 * c)
                ));

                let mut data = new(self.bit_width, self.num_vertices * self.num_chunks);
                match self.par_solve(
                    &mut pl.concurrent(),
                    pl,
                    chunk_store.iter().unwrap(),
                    &mut data,
                    new,
                    Self::peel_chunk,
                ) {
                    Ok(()) => break data,
                    Err(_) => {
                        if dup_count >= 3 {
                            bail!("Duplicate keys (duplicate 128-bit signatures with four different seeds");
                        }
                        warn!("Duplicate 128-bit signature, trying again...");
                        dup_count += 1;
                        continue;
                    }
                }
            } else {
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

                //(chunk_high_bits, max_num_threads) = compute_params(num_keys, SHARDED);
                (self.chunk_high_bits, max_num_threads) = (8, 8);
                self.log2_seg_size = comp_log2_seg_size(3, self.num_keys >> self.chunk_high_bits);
                dbg!(self.log2_seg_size);
                c = 1.10; // size_factor(3, num_keys);
                dbg!(c, self.log2_seg_size);

                self.num_chunks = 1 << self.chunk_high_bits;
                self.chunk_mask = (1u32 << self.chunk_high_bits) - 1;

                pl.start("Sorting...");
                sig_vals.radix_sort_unstable();
                pl.done_with_count(self.num_keys);

                pl.start("Checking for duplicates...");

                let mut chunk_sizes = vec![0_usize; self.num_chunks];
                let mut dup = false;

                chunk_sizes[sig_vals[0].sig.chunk(self.chunk_high_bits, self.chunk_mask)] += 1;

                for w in sig_vals.windows(2) {
                    chunk_sizes[w[1].sig.chunk(self.chunk_high_bits, self.chunk_mask)] += 1;
                    if w[0].sig == w[1].sig {
                        dup = true;
                        break;
                    }
                }

                pl.done_with_count(self.num_keys);

                if dup {
                    if dup_count >= 3 {
                        bail!("Duplicate keys (duplicate 128-bit signatures with four different seeds)");
                    }
                    warn!("Duplicate 128-bit signature, trying again...");
                    dup_count += 1;
                    continue;
                }

                self.bit_width = max_value.len() as usize;
                pl.info(format_args!(
                    "max value = {}, bit width = {}",
                    max_value, self.bit_width
                ));

                self.l = (((self.num_keys >> self.chunk_high_bits) as f64 * c).ceil() as usize)
                    .div_ceil(1 << self.log2_seg_size);
                let num_vertices = (1 << self.log2_seg_size) * (self.l + 2);
                dbg!(num_vertices);

                pl.info(format_args!(
                    "Size {:.2}%",
                    (100.0 * (num_vertices * self.num_chunks) as f64) / (self.num_keys as f64 * c)
                ));

                let mut data = new(self.bit_width, num_vertices * self.num_chunks);

                if num_vertices < 64000000 {
                    eprintln!("Linear case");

                    match self.par_solve(
                        &mut pl.concurrent(),
                        pl,
                        sig_vals
                            .arbitrary_chunks(&chunk_sizes)
                            .map(Cow::Borrowed)
                            .enumerate(),
                        &mut data,
                        new,
                        Self::peel_chunk,
                    ) {
                        Ok(()) => break data,
                        Err(_) => continue,
                    }
                }

                if SHARDED {
                    eprintln!("*****************");
                    match self.par_solve(
                        &mut pl.concurrent(),
                        pl,
                        sig_vals
                            .arbitrary_chunks(&chunk_sizes)
                            .map(Cow::Borrowed)
                            .enumerate(),
                        &mut data,
                        new,
                        Self::peel_chunk,
                    ) {
                        Ok(()) => break data,
                        Err(_) => {}
                    }
                } else {
                    match self.peel_chunk(pl, no_logging![], 0, sig_vals.into(), data) {
                        Ok((_, data)) => break data,
                        Err(_) => {}
                    }
                }
            }

            seed += 1;
        };

        pl.info(format_args!(
            "bits/keys: {}",
            data.len() as f64 * self.bit_width as f64 / self.num_keys as f64
        ));

        Ok(VFunc {
            seed,
            l: self.l,
            high_bits: self.chunk_high_bits,
            chunk_mask: self.chunk_mask,
            num_keys: self.num_keys,
            log2_seg_size: self.log2_seg_size,
            data,
            _marker_t: std::marker::PhantomData,
            _marker_os: std::marker::PhantomData,
        })
    }

    fn solve_lin<'a>(
        &self,
        _main_pl: &mut impl ProgressLog,
        pl: &mut impl ProgressLog,
        chunk_index: usize,
        mut chunk: Cow<'a, [SigVal<O>]>,
        mut data: D,
    ) -> anyhow::Result<(usize, D)> {
        if let Cow::Owned(chunk) = &mut chunk {
            chunk.radix_sort_unstable();
        }

        if chunk.par_windows(2).any(|w| w[0].sig == w[1].sig) {
            bail!("Duplicate signature in chunk");
        }

        pl.start(format!(
            "Generating system for chunk {}/{}...",
            chunk_index + 1,
            self.num_chunks
        ));
        let mut system = Modulo2System::<O>::new(self.num_vertices);
        chunk.iter().for_each(|sig_val| {
            let mut eq = Modulo2Equation::<O>::new(sig_val.val, self.num_vertices);
            for &v in sig_val
                .sig
                .edge(self.chunk_high_bits, self.l, self.log2_seg_size)
                .iter()
            {
                eq.add(v);
            }
            system.add(eq);
        });
        pl.done_with_count(chunk.len());

        pl.start("Solving system...");
        let result = system.lazy_gaussian_elimination_constructor()?;
        pl.done();

        for (v, &value) in result.iter().enumerate() {
            data.set(v, value);
        }

        Ok((chunk_index, data))
    }
}
