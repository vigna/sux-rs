/*
*
* SPDX-FileCopyrightText: 2023 Sebastiano Vigna
*
* SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
*/

// We import selectively to be able to use AtomicHelper
use crate::bits::*;
use crate::prelude::ConvertTo;
use crate::traits::bit_field_slice;
use crate::utils::*;
use anyhow::bail;
use arbitrary_chunks::ArbitraryChunks;
use bit_field_slice::AtomicBitFieldSlice;
use bit_field_slice::BitFieldSlice;
use bit_field_slice::Word;
use common_traits::Atomic;
use common_traits::{AsBytes, AtomicUnsignedInt, IntoAtomic};
use derivative::Derivative;
use derive_setters::*;
use dsi_progress_logger::*;
use epserde::prelude::*;
use log::warn;
use mem_dbg::*;
use rayon::prelude::*;
use rdst::*;
use std::borrow::Cow;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::thread;
use Ordering::Relaxed;

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

/**

An edge list represented by a 64-bit integer. The lower SIDE_SHIFT bits
contain a XOR of the edges; the next two bits contain a XOR of the side
of the edges; the remaining DEG_SHIFT the upper bits contain the degree.

The degree can be stored with a small number of bits because the
graph is random, so the maximum degree is O(log log n).

This approach reduces the core memory usage for the hypergraph to
a `u64` per vertex. Edges are derived on the fly from the 128-bit signatures.

*/

#[derive(Debug, Default)]
struct EdgeList(u64);
impl EdgeList {
    const DEG_SHIFT: u32 = u64::BITS as u32 - 16;
    const SIDE_SHIFT: u32 = u64::BITS as u32 - 18;
    const SIDE_MASK: usize = 3;
    const EDGE_INDEX_MASK: u64 = (1_u64 << Self::SIDE_SHIFT) - 1;
    const DEG: u64 = 1_u64 << Self::DEG_SHIFT;
    const MAX_DEG: usize = (u64::MAX >> Self::DEG_SHIFT) as usize;

    fn store(&mut self, edge: usize, side: usize) {
        debug_assert!(side as u64 <= Self::EDGE_INDEX_MASK);
        *self = EdgeList((side as u64) << Self::SIDE_SHIFT | edge as u64);
    }

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
    fn edge_index(&self) -> usize {
        debug_assert!(self.degree() == 0);
        (self.0 & Self::EDGE_INDEX_MASK) as usize
    }

    #[inline(always)]
    fn edge_index_side(&self) -> (usize, usize) {
        (
            (self.0 & EdgeList::EDGE_INDEX_MASK) as usize,
            (self.0 >> Self::SIDE_SHIFT) as usize & Self::SIDE_MASK,
        )
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

#[inline(always)]
#[must_use]
fn chunk(sig: &[u64; 2], chunk_high_bits: u32, chunk_mask: u32) -> usize {
    (sig[0].rotate_left(chunk_high_bits) & chunk_mask as u64) as usize
}

#[inline(always)]
#[must_use]
fn edge(sig: &[u64; 2], chunk_high_bits: u32, l: usize, log2_seg_size: u32) -> [usize; 3] {
    let first_segment = ((sig[0] << chunk_high_bits >> 32) * l as u64 >> 32) as usize;
    let start = first_segment << log2_seg_size;
    let segment_size = 1 << log2_seg_size;
    let segment_mask = segment_size - 1;

    [
        (sig[0] as usize & segment_mask) + start,
        ((sig[1] >> 32) as usize & segment_mask) + start + segment_size,
        (sig[1] as usize & segment_mask) + start + 2 * segment_size,
    ]
}

/**

A builder for [`VFunc`].

Keys must implement the [`ToSig`] trait, which provides a method to
compute a 128-bit signature of the key.

The output type `O` can be selected to be any of the unsigned integer types
with an atomic counterpart; the default is `usize`.

There are two construction modes: in core memory (default) and [offline](VFuncBuilder::offline). In the first case,
space will be allocated in core memory for 128-bit signatures and associated values for all keys;
in the second case, such information will be stored in a number of on-disk buckets.

Once signatures have been computed, each parallel thread will process a chunk of the signatures;
for each signature in a chunk, a thread will allocate about two `usize` values in core memory;
in the case of offline construction, also signatures and values in the chunk
will be stored in core memory.

For very large key sets chunks will be significantly smaller than the number of keys,
so the memory usage, in particular in offline mode, can be significantly reduced. Note that
using too many threads might actually be harmful due to memory contention: eight is usually a good value.

 */
#[derive(Setters, Debug, Derivative)]
#[derivative(Default)]
#[setters(generate = false)]
pub struct VFuncBuilder<
    T: ?Sized + ToSig,
    O: ZeroCopy + Word + IntoAtomic,
    D: bit_field_slice::BitFieldSlice<O> = BitFieldVec<O>,
> {
    #[setters(generate = true)]
    /// The number of parallel threads to use.
    num_threads: usize,
    #[setters(generate = true)]
    /// Use disk-based buckets to reduce core memory usage at construction time.
    offline: bool,
    /// The base-2 logarithm of the number of buckets. Used only if `offline` is `true`. The default is 8.
    #[setters(generate = true, strip_option)]
    log2_buckets: Option<u32>,
    _marker_t: PhantomData<T>,
    _marker_od: PhantomData<(O, D)>,
}

/**

Static functions with 10%-11% space overhead for large key sets,
fast parallel construction, and fast queries.

Instances of this structure are immutable; they are built using a [`VFuncBuilder`],
and can be serialized using [ε-serde](`epserde`).

*/

#[derive(Epserde, Debug, MemDbg, MemSize)]
pub struct VFunc<
    T: ?Sized + ToSig,
    O: ZeroCopy + Word + IntoAtomic = usize,
    D: bit_field_slice::BitFieldSlice<O> = BitFieldVec<O>,
> {
    high_bits: u32,
    log2_segment_size: u32,
    chunk_mask: u32,
    seed: u64,
    l: usize,
    num_keys: usize,
    data: D,
    _marker_t: std::marker::PhantomData<T>,
    _marker_o: std::marker::PhantomData<O>,
}

fn compute_params(num_keys: usize, pl: &mut impl ProgressLog) -> (u32, usize) {
    let eps = 0.001;
    let chunk_high_bits = {
        let t = (num_keys as f64 * eps * eps / 2.0).ln();

        if t > 0.0 {
            ((t - t.ln()) / 2_f64.ln()).ceil()
        } else {
            0.0
        }
    }
    .min(10.0) as u32; // More than 1000 chunks make no sense
    let max_num_threads = 1; //1.max((1 << chunk_high_bits) / 8).min(8); TODO

    dbg!((chunk_high_bits, max_num_threads))
}

enum ParSolveResult<O: Word + IntoAtomic, D: AtomicBitFieldSlice<O>>
where
    O::AtomicType: AtomicUnsignedInt + AsBytes,
{
    DuplicateSignature,
    CantPeel,
    Ok(D, PhantomData<O>),
}

#[allow(clippy::too_many_arguments)]
fn par_solve<
    'a,
    O: Word + ZeroCopy + Send + Sync + IntoAtomic + 'static,
    I: Iterator<Item = (usize, Cow<'a, [SigVal<O>]>)> + Send,
    D: AtomicBitFieldSlice<O> + Send + Sync,
>(
    chunk_iter: I,
    data: D,
    num_chunks: usize,
    num_vertices: usize,
    num_threads: usize,
    log2_segment_size: u32,
    l: usize,
    main_pl: &mut (impl ProgressLog + Send),
) -> ParSolveResult<O, D>
where
    O::AtomicType: AtomicUnsignedInt + AsBytes,
{
    const SIDE_PERM: [usize; 4] = [1, 2, 0, 1];
    let chunk_high_bits = num_chunks.ilog2();
    let chunk_iter = std::sync::Arc::new(Mutex::new(chunk_iter));
    let failed_peeling = AtomicBool::new(false);
    let duplicate_signature = AtomicBool::new(false);
    main_pl.info(format_args!("Using {} threads", num_threads));
    main_pl
        .item_name("chunk")
        .expected_updates(Some(num_chunks));
    main_pl.start("Analyzing chunks...");
    let main_pl = std::sync::Arc::new(Mutex::new(main_pl));
    thread::scope(|s| {
        for _ in 0..num_threads {
            s.spawn(|| {
                loop {
                    if failed_peeling.load(Relaxed) || duplicate_signature.load(Relaxed) {
                        return;
                    }
                    let (chunk_index, mut chunk) = match chunk_iter.lock().unwrap().next() {
                        None => return,
                        Some((chunk_index, chunk)) => (chunk_index, chunk),
                    };

                    if let Cow::Owned(chunk) = &mut chunk {
                        chunk.radix_sort_unstable();
                    }

                    if chunk.par_windows(2).any(|w| w[0].sig == w[1].sig) {
                        duplicate_signature.store(true, Ordering::Relaxed);
                        return;
                    }

                    let mut pl = main_pl.lock().unwrap().clone();
                    pl.item_name("edge");
                    pl.start(format!(
                        "Generating graph for chunk {}/{}...",
                        chunk_index + 1,
                        num_chunks
                    ));
                    let mut edge_lists = Vec::new();
                    edge_lists.resize_with(num_vertices, EdgeList::default);
                    chunk.iter().enumerate().for_each(|(edge_index, sig_val)| {
                        for (side, &v) in edge(&sig_val.sig, chunk_high_bits, l, log2_segment_size)
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
                        num_chunks
                    ));
                    let mut stack = Vec::new();
                    for v in (0..num_vertices).rev() {
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
                            stack[curr] = v;
                            curr += 1;
                            let [u0, u1, u2] = edge(
                                &chunk[edge_index].sig,
                                chunk_high_bits,
                                l,
                                log2_segment_size,
                            );
                            let perm = [u1, u2, u0, u1];

                            let u = perm[side];
                            edge_lists[u].remove(edge_index, SIDE_PERM[side]);
                            if edge_lists[u].degree() == 1 {
                                stack.push(u);
                            }

                            let u = perm[side + 1];
                            edge_lists[u].remove(edge_index, SIDE_PERM[side + 1]);
                            if edge_lists[u].degree() == 1 {
                                stack.push(u);
                            }

                            edge_lists[v].store(edge_index, side);
                        }
                        stack.truncate(curr);
                    }
                    if chunk.len() != stack.len() {
                        warn!(
                            "Peeling failed for chunk {}/{}",
                            chunk_index + 1,
                            num_chunks
                        );
                        failed_peeling.store(true, Ordering::Relaxed);
                        return;
                    }
                    pl.done_with_count(chunk.len());

                    pl.start(format!(
                        "Assigning values for chunk {}/{}...",
                        chunk_index + 1,
                        num_chunks
                    ));
                    while let Some(mut v) = stack.pop() {
                        let edge_index = edge_lists[v].edge_index();
                        let mut edge = edge(
                            &chunk[edge_index].sig,
                            chunk_high_bits,
                            l,
                            log2_segment_size,
                        );
                        let chunk_offset = chunk_index * num_vertices;
                        v += chunk_offset;
                        edge.iter_mut().for_each(|v| {
                            *v += chunk_offset;
                        });
                        let value = if v == edge[0] {
                            data.get_atomic(edge[1], Relaxed) ^ data.get_atomic(edge[2], Relaxed)
                        } else if v == edge[1] {
                            data.get_atomic(edge[0], Relaxed) ^ data.get_atomic(edge[2], Relaxed)
                        } else {
                            data.get_atomic(edge[0], Relaxed) ^ data.get_atomic(edge[1], Relaxed)
                        };

                        data.set_atomic(v, chunk[edge_index].val ^ value, Relaxed);
                        debug_assert_eq!(
                            data.get_atomic(edge[0], Relaxed)
                                ^ data.get_atomic(edge[1], Relaxed)
                                ^ data.get_atomic(edge[2], Relaxed),
                            chunk[edge_index].val
                        );
                    }
                    pl.done_with_count(chunk.len());

                    pl.start(format!(
                        "Completed chunk {}/{}.",
                        chunk_index + 1,
                        num_chunks
                    ));
                    main_pl.lock().unwrap().update_and_display();
                }
            });
        }
    });

    if failed_peeling.load(Relaxed) {
        ParSolveResult::CantPeel
    } else if duplicate_signature.load(Relaxed) {
        ParSolveResult::DuplicateSignature
    } else {
        main_pl.lock().unwrap().done();
        ParSolveResult::Ok(data, PhantomData)
    }
}

impl<T: ?Sized + ToSig, O: ZeroCopy + Word + IntoAtomic, D: bit_field_slice::BitFieldSlice<O>>
    VFunc<T, O, D>
where
    O::AtomicType: AtomicUnsignedInt + AsBytes,
{
    /// Return the value associated with the given signature.
    ///
    /// This method is mainly useful in the construction of compound functions.
    #[inline(always)]
    pub fn get_by_sig(&self, sig: &[u64; 2]) -> O {
        let edge = edge(sig, self.high_bits, self.l, self.log2_segment_size);
        let chunk = chunk(sig, self.high_bits, self.chunk_mask);
        // chunk * self.segment_size * (2^log2_l + 2)
        let chunk_offset = chunk * ((self.l + 2) << self.log2_segment_size);

        unsafe {
            self.data.get_unchecked(edge[0] + chunk_offset)
                ^ self.data.get_unchecked(edge[1] + chunk_offset)
                ^ self.data.get_unchecked(edge[2] + chunk_offset)
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

impl<T: ?Sized + ToSig, O: ZeroCopy + Word + IntoAtomic + 'static> VFuncBuilder<T, O, Vec<O>>
where
    O::AtomicType: AtomicUnsignedInt + AsBytes,
    Vec<O>: BitFieldSlice<O>,
    Vec<O::AtomicType>: AtomicBitFieldSlice<O> + ConvertTo<Vec<O>>,
{
    pub fn build(
        self,
        into_keys: impl RewindableIOLender<T>,
        into_values: impl RewindableIOLender<O>,
        pl: &mut (impl ProgressLog + Send),
    ) -> anyhow::Result<VFunc<T, O, Vec<O>>> {
        self._build::<Vec<O::AtomicType>>(
            into_keys,
            into_values,
            |_bit_width: usize, len: usize| (0..len).map(|_| O::AtomicType::new(O::ZERO)).collect(),
            pl,
        )
    }
}

impl<T: ?Sized + ToSig, O: ZeroCopy + Word + IntoAtomic + 'static>
    VFuncBuilder<T, O, BitFieldVec<O>>
where
    O::AtomicType: AtomicUnsignedInt + AsBytes,
    Vec<O::AtomicType>: AtomicBitFieldSlice<O> + ConvertTo<Vec<O>>,
{
    pub fn build(
        self,
        into_keys: impl RewindableIOLender<T>,
        into_values: impl RewindableIOLender<O>,
        pl: &mut (impl ProgressLog + Send),
    ) -> anyhow::Result<VFunc<T, O, BitFieldVec<O>>> {
        self._build::<AtomicBitFieldVec<O>>(
            into_keys,
            into_values,
            |bit_width: usize, len: usize| AtomicBitFieldVec::<O>::new(bit_width, len),
            pl,
        )
    }
}

impl<
        T: ?Sized + ToSig,
        O: ZeroCopy + Word + IntoAtomic + 'static,
        D: bit_field_slice::BitFieldSlice<O>,
    > VFuncBuilder<T, O, D>
where
    O::AtomicType: AtomicUnsignedInt + AsBytes,
{
    /// Build and return a new function with given keys and values.
    fn _build<A: AtomicBitFieldSlice<O> + Send + Sync>(
        self,
        mut into_keys: impl RewindableIOLender<T>,
        mut into_values: impl RewindableIOLender<O>,
        new: fn(usize, usize) -> A,
        pl: &mut (impl ProgressLog + Send),
    ) -> anyhow::Result<VFunc<T, O, D>>
    where
        A: ConvertTo<D>,
    {
        // Loop until success or duplicate detection
        let mut dup_count = 0;
        let mut seed = 0;
        let (
            mut num_keys,
            mut bit_width,
            mut log2_seg_size,
            mut chunk_high_bits,
            mut chunk_mask,
            mut l,
        );
        let data = loop {
            pl.item_name("key");
            pl.start("Reading input...");
            let mut max_value = O::ZERO;
            //let mut chunk_sizes;
            let (mut max_num_threads, c);

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
                num_keys = sig_sorter.len();
                pl.done();

                (chunk_high_bits, max_num_threads) = compute_params(num_keys, pl);
                chunk_high_bits = 0;
                max_num_threads = 1;
                log2_seg_size = comp_log2_seg_size(3, num_keys);
                c = size_factor(3, num_keys);
                dbg!(c, log2_seg_size);

                let num_chunks = 1 << chunk_high_bits;
                chunk_mask = (1u32 << chunk_high_bits) - 1;

                let mut chunk_store = sig_sorter.into_chunk_store(chunk_high_bits)?;
                let chunk_sizes = chunk_store.chunk_sizes();

                bit_width = max_value.len() as usize;
                pl.info(format_args!(
                    "max value = {}, bit width = {}",
                    max_value, bit_width
                ));

                l = ((num_keys as f64 * c).ceil() as usize).div_ceil(1 << log2_seg_size);
                let num_vertices = (1 << log2_seg_size) * (l + 2);
                pl.info(format_args!(
                    "Size {:.2}%",
                    (100.0 * (num_vertices * num_chunks) as f64) / (num_keys as f64 * c)
                ));

                match par_solve(
                    chunk_store.iter().unwrap(),
                    new(bit_width, num_vertices * num_chunks),
                    num_chunks,
                    num_vertices,
                    match self.num_threads {
                        0 => max_num_threads,
                        _ => self.num_threads,
                    },
                    log2_seg_size,
                    l,
                    pl,
                ) {
                    ParSolveResult::DuplicateSignature => {
                        if dup_count >= 3 {
                            bail!("Duplicate keys (duplicate 128-bit signatures with four different seeds");
                        }
                        warn!("Duplicate 128-bit signature, trying again...");
                        dup_count += 1;
                        continue;
                    }
                    ParSolveResult::CantPeel => {}
                    ParSolveResult::Ok(data, PhantomData) => break data,
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
                num_keys = sig_vals.len();

                (chunk_high_bits, max_num_threads) = compute_params(num_keys, pl);
                chunk_high_bits = 0;
                max_num_threads = 1;
                log2_seg_size = comp_log2_seg_size(3, num_keys);
                c = size_factor(3, num_keys);
                dbg!(c, log2_seg_size);

                let num_chunks = 1 << chunk_high_bits;
                chunk_mask = (1u32 << chunk_high_bits) - 1;

                pl.start("Sorting...");
                sig_vals.radix_sort_unstable();
                pl.done_with_count(num_keys);

                pl.start("Checking for duplicates...");

                let mut chunk_sizes = vec![0_usize; num_chunks];
                let mut dup = false;

                chunk_sizes[chunk(&sig_vals[0].sig, chunk_high_bits, chunk_mask)] += 1;

                for w in sig_vals.windows(2) {
                    assert!(w[0].sig[0] <= w[1].sig[0]);
                    chunk_sizes[chunk(&w[1].sig, chunk_high_bits, chunk_mask)] += 1;
                    if w[0].sig == w[1].sig {
                        dup = true;
                        break;
                    }
                }

                pl.done_with_count(num_keys);

                if dup {
                    if dup_count >= 3 {
                        bail!("Duplicate keys (duplicate 128-bit signatures with four different seeds)");
                    }
                    warn!("Duplicate 128-bit signature, trying again...");
                    dup_count += 1;
                    continue;
                }

                bit_width = max_value.len() as usize;
                pl.info(format_args!(
                    "max value = {}, bit width = {}",
                    max_value, bit_width
                ));

                l = ((num_keys as f64 * c).ceil() as usize).div_ceil(1 << log2_seg_size);
                let num_vertices = (1 << log2_seg_size) * (l + 2);
                pl.info(format_args!(
                    "Size {:.2}%",
                    (100.0 * (num_vertices * num_chunks) as f64) / (num_keys as f64 * c)
                ));

                match par_solve(
                    sig_vals
                        .arbitrary_chunks(&chunk_sizes)
                        .map(Cow::Borrowed)
                        .enumerate(),
                    new(bit_width, num_vertices * num_chunks),
                    num_chunks,
                    num_vertices,
                    match self.num_threads {
                        0 => max_num_threads,
                        _ => self.num_threads,
                    },
                    log2_seg_size,
                    l,
                    pl,
                ) {
                    ParSolveResult::DuplicateSignature => {
                        unreachable!("Already checked for duplicates")
                    }
                    ParSolveResult::CantPeel => {}
                    ParSolveResult::Ok(data, PhantomData) => break data,
                }
            }

            seed += 1;
        };

        pl.info(format_args!(
            "bits/keys: {}",
            0 //data.len() as f64 * bit_width as f64 / num_keys as f64
        ));

        Ok(VFunc {
            seed,
            l,
            high_bits: chunk_high_bits,
            chunk_mask,
            num_keys,
            log2_segment_size: log2_seg_size,
            data: data.convert_to().unwrap(),
            _marker_t: std::marker::PhantomData,
            _marker_o: std::marker::PhantomData,
        })
    }
}
