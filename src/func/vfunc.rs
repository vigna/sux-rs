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
use bit_field_slice::BitFieldSliceCore;
use bit_field_slice::Word;
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
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::thread;
use Ordering::Relaxed;

const PARAMS: [(usize, u32, f64); 15] = [
    (0, 0, 1.23),
    (10000, 5, 1.23),
    (12500, 5, 1.22),
    (19531, 5, 1.21),
    (30516, 5, 1.20),
    (38145, 6, 1.19),
    (47681, 6, 1.18),
    (74501, 6, 1.17),
    (116407, 6, 1.16),
    (227356, 6, 1.15),
    (355243, 7, 1.14),
    (693832, 7, 1.13),
    (2117406, 7, 1.12),
    (6461807, 8, 1.11),
    (60180252, 9, 1.10),
];

/**

An edge list represented by a 64-bit integer. The lower DEG_SHIFT bits
contian a XOR of the edges, while the upper bits contain the degree.

The degree can be stored with a small number of bits because the
graph is random, so the maximum degree is O(log log n).

This approach reduces the core memory usage for the hypergraph to
a `usize` per vertex. Edges are derived on the fly from the 128-bit signatures.

*/

#[derive(Debug, Clone, MemDbg, MemSize, Default)]
struct EdgeList(usize);
impl EdgeList {
    const DEG_SHIFT: usize = usize::BITS as usize - 16;
    const EDGE_INDEX_MASK: usize = (1_usize << EdgeList::DEG_SHIFT) - 1;
    const DEG: usize = 1_usize << EdgeList::DEG_SHIFT;
    const MAX_DEG: usize = usize::MAX >> EdgeList::DEG_SHIFT;

    #[inline(always)]
    fn add(&mut self, edge: usize) {
        debug_assert!(self.degree() < Self::MAX_DEG);
        self.0 += EdgeList::DEG;
        self.0 ^= edge;
    }

    #[inline(always)]
    fn remove(&mut self, edge: usize) {
        debug_assert!(self.degree() > 0);
        self.0 -= EdgeList::DEG;
        self.0 ^= edge;
    }

    #[inline(always)]
    fn degree(&self) -> usize {
        self.0 >> EdgeList::DEG_SHIFT
    }

    /// Return the edge index after a call to `zero`.
    #[inline(always)]
    fn edge_index(&self) -> usize {
        debug_assert!(self.degree() == 0);
        self.0 & EdgeList::EDGE_INDEX_MASK
    }

    #[inline(always)]
    /// Set the degree of a list of degree one to zero,
    /// leaving the edge index unchanged.
    fn zero(&mut self) {
        debug_assert!(self.degree() == 1);
        self.0 -= EdgeList::DEG;
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
fn edge(sig: &[u64; 2], log2_l: u32, segment_size: usize) -> [usize; 3] {
    let first_segment = (sig[0] >> 32 & ((1 << log2_l) - 1)) as usize;
    let start = first_segment * segment_size;
    [
        (((sig[0] & 0xFFFFFFFF) * segment_size as u64) >> 32) as usize + start,
        (((sig[1] >> 32) * segment_size as u64) >> 32) as usize + start + segment_size,
        (((sig[1] & 0xFFFFFFFF) * segment_size as u64) >> 32) as usize + start + 2 * segment_size,
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
    O: ZeroCopy + SerializeInner + DeserializeInner + Word + IntoAtomic,
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
    _marker_t: std::marker::PhantomData<T>,
    _marker_o: std::marker::PhantomData<O>,
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
    O: ZeroCopy + SerializeInner + DeserializeInner + Word + IntoAtomic = usize,
    S: bit_field_slice::BitFieldSlice<O> = BitFieldVec<O>,
> {
    seed: u64,
    log2_l: u32,
    high_bits: u32,
    chunk_mask: u32,
    num_keys: usize,
    segment_size: usize,
    values: S,
    _marker_t: std::marker::PhantomData<T>,
    _marker_o: std::marker::PhantomData<O>,
}

fn compute_params(num_keys: usize, pl: &mut impl ProgressLog) -> (u32, usize, u32, f64) {
    let (chunk_high_bits, max_num_threads, log2_l, c);

    if num_keys < PARAMS[PARAMS.len() - 2].0 {
        // Too few keys, we cannot split into chunks
        chunk_high_bits = 0;
        max_num_threads = 1;
        (_, log2_l, c) = PARAMS
            .iter()
            .rev()
            .filter(|(n, _l, _c)| n <= &num_keys)
            .copied()
            .next()
            .unwrap(); // first n is 0, so this is always valid
    } else if num_keys < PARAMS[PARAMS.len() - 1].0 {
        // We are in the 1.11 regime. We can reduce memory
        // usage by doing single thread.
        chunk_high_bits = (num_keys / PARAMS[PARAMS.len() - 2].0).ilog2();
        max_num_threads = 1;
        (_, log2_l, c) = PARAMS[PARAMS.len() - 2];
    } else {
        // We are in the 1.10 regime. We can increase the number
        // of threads, but we need to be careful not to use too
        // much memory.

        let eps = 0.001;
        chunk_high_bits = {
                    let t = (num_keys as f64 * eps * eps / 2.0).ln();

                    if t > 0.0 {
                        ((t - t.ln()) / 2_f64.ln()).ceil()
                    } else {
                        0.0
                    }
                }
                .min(10.0) // More than 1000 chunks make no sense
                .min((num_keys / PARAMS[PARAMS.len() - 1].0).ilog2() as f64) // Let's keep the chunks in the 1.10 regime
                    as u32;
        // By default no more than 8 threads
        max_num_threads = 1.max((1 << chunk_high_bits) / 8).min(8);
        (_, log2_l, c) = PARAMS[PARAMS.len() - 1];
    }

    pl.info(format_args!(
        "chunk high bits = {}, l = {}, c = {}",
        chunk_high_bits,
        1 << log2_l,
        c
    ));

    (chunk_high_bits, max_num_threads, log2_l, c)
}

enum ParSolveResult<O: Word + IntoAtomic> {
    DuplicateSignature,
    CantPeel,
    Ok(AtomicBitFieldVec<O>),
}

#[allow(clippy::too_many_arguments)]
fn par_solve<
    'a,
    O: Word + ZeroCopy + Send + Sync + IntoAtomic + 'static,
    I: Iterator<Item = (usize, Cow<'a, [SigVal<O>]>)> + Send,
>(
    chunk_iter: I,
    bit_width: usize,
    num_chunks: usize,
    num_vertices: usize,
    num_threads: usize,
    segment_size: usize,
    log2_l: u32,
    main_pl: &mut (impl ProgressLog + Send),
) -> ParSolveResult<O>
where
    O::AtomicType: AtomicUnsignedInt + AsBytes,
{
    use crate::traits::bit_field_slice::AtomicHelper;
    let data = AtomicBitFieldVec::<O>::new(bit_width, num_vertices * num_chunks);
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
            s.spawn(|| loop {
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
                    for &v in edge(&sig_val.sig, log2_l, segment_size).iter() {
                        edge_lists[v].add(edge_index);
                    }
                });
                pl.done_with_count(chunk.len());

                pl.start(format!(
                    "Peeling graph for chunk {}/{}...",
                    chunk_index + 1,
                    num_chunks
                ));
                let mut stack = Vec::new();
                for v in 0..num_vertices {
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
                        edge_lists[v].zero();
                        let edge_index = edge_lists[v].edge_index();
                        stack[curr] = v;
                        curr += 1;
                        // Degree is necessarily 0
                        for &x in edge(&chunk[edge_index].sig, log2_l, segment_size).iter() {
                            if x != v {
                                edge_lists[x].remove(edge_index);
                                if edge_lists[x].degree() == 1 {
                                    stack.push(x);
                                }
                            }
                        }
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
                    let mut edge = edge(&chunk[edge_index].sig, log2_l, segment_size);
                    let chunk_offset = chunk_index * num_vertices;
                    v += chunk_offset;
                    edge.iter_mut().for_each(|v| {
                        *v += chunk_offset;
                    });
                    let value = if v == edge[0] {
                        data.get(edge[1], Relaxed) ^ data.get(edge[2], Relaxed)
                    } else if v == edge[1] {
                        data.get(edge[0], Relaxed) ^ data.get(edge[2], Relaxed)
                    } else {
                        data.get(edge[0], Relaxed) ^ data.get(edge[1], Relaxed)
                    };

                    data.set(v, chunk[edge_index].val ^ value, Relaxed);
                    debug_assert_eq!(
                        data.get(edge[0], Relaxed)
                            ^ data.get(edge[1], Relaxed)
                            ^ data.get(edge[2], Relaxed),
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
            });
        }
    });

    if failed_peeling.load(Relaxed) {
        ParSolveResult::CantPeel
    } else if duplicate_signature.load(Relaxed) {
        ParSolveResult::DuplicateSignature
    } else {
        main_pl.lock().unwrap().done();
        ParSolveResult::Ok(data)
    }
}

impl<
        T: ?Sized + ToSig,
        O: ZeroCopy + SerializeInner + DeserializeInner + Word + IntoAtomic,
        S: bit_field_slice::BitFieldSlice<O>,
    > VFunc<T, O, S>
where
    O::AtomicType: AtomicUnsignedInt + AsBytes,
    BitFieldVec<O>: From<AtomicBitFieldVec<O, Vec<O::AtomicType>>>,
{
    /// Return the value associated with the given signature.
    ///
    /// This method is mainly useful in the construction of compound functions.
    pub fn get_by_sig(&self, sig: &[u64; 2]) -> O {
        let edge = edge(sig, self.log2_l, self.segment_size);
        let chunk = chunk(sig, self.high_bits, self.chunk_mask);
        // chunk * self.segment_size * (2^log2_l + 2)
        let chunk_offset = chunk * ((self.segment_size << self.log2_l) + (self.segment_size << 1));

        unsafe {
            self.values.get_unchecked(edge[0] + chunk_offset)
                ^ self.values.get_unchecked(edge[1] + chunk_offset)
                ^ self.values.get_unchecked(edge[2] + chunk_offset)
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

impl<
        T: ?Sized + ToSig,
        O: ZeroCopy + SerializeInner + DeserializeInner + Word + IntoAtomic + 'static,
    > VFuncBuilder<T, O>
where
    O::AtomicType: AtomicUnsignedInt + AsBytes,
    BitFieldVec<O>: From<AtomicBitFieldVec<O, Vec<O::AtomicType>>>,
{
    /// Build and return a new function with given keys and values.
    pub fn build(
        self,
        mut into_keys: impl RewindableIOLender<T>,
        mut into_values: impl RewindableIOLender<O>,
        pl: &mut (impl ProgressLog + Send),
    ) -> anyhow::Result<VFunc<T, O>> {
        // Loop until success or duplicate detection
        let mut dup_count = 0;
        let mut seed = 0;
        let (
            mut num_keys,
            mut bit_width,
            mut segment_size,
            mut chunk_high_bits,
            mut chunk_mask,
            mut log2_l,
        );
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
                num_keys = sig_sorter.len();
                pl.done();

                (chunk_high_bits, max_num_threads, log2_l, c) = compute_params(num_keys, pl);

                let num_chunks = 1 << chunk_high_bits;
                chunk_mask = (1u32 << chunk_high_bits) - 1;

                let mut chunk_store = sig_sorter.into_chunk_store(chunk_high_bits)?;
                let chunk_sizes = chunk_store.chunk_sizes();

                bit_width = max_value.len() as usize;
                pl.info(format_args!(
                    "max value = {}, bit width = {}",
                    max_value, bit_width
                ));

                let l = 1 << log2_l;
                segment_size =
                    ((*chunk_sizes.iter().max().unwrap() as f64 * c).ceil() as usize + l + 1)
                        / (l + 2);
                let num_vertices = segment_size * (l + 2);
                pl.info(format_args!(
                    "Size {:.2}%",
                    (100.0 * (num_vertices * num_chunks) as f64) / (num_keys as f64 * c)
                ));

                match par_solve(
                    chunk_store.iter().unwrap(),
                    bit_width,
                    num_chunks,
                    num_vertices,
                    match self.num_threads {
                        0 => max_num_threads,
                        _ => self.num_threads,
                    },
                    segment_size,
                    log2_l,
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
                    ParSolveResult::Ok(data) => break data,
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

                (chunk_high_bits, max_num_threads, log2_l, c) = compute_params(num_keys, pl);

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

                let l = 1 << log2_l;
                segment_size =
                    ((*chunk_sizes.iter().max().unwrap() as f64 * c).ceil() as usize + l + 1)
                        / (l + 2);
                let num_vertices = segment_size * (l + 2);
                pl.info(format_args!(
                    "Size {:.2}%",
                    (100.0 * (num_vertices * num_chunks) as f64) / (sig_vals.len() as f64 * c)
                ));

                match par_solve(
                    sig_vals
                        .arbitrary_chunks(&chunk_sizes)
                        .map(Cow::Borrowed)
                        .enumerate(),
                    bit_width,
                    num_chunks,
                    num_vertices,
                    match self.num_threads {
                        0 => max_num_threads,
                        _ => self.num_threads,
                    },
                    segment_size,
                    log2_l,
                    pl,
                ) {
                    ParSolveResult::DuplicateSignature => {
                        unreachable!("Already checked for duplicates")
                    }
                    ParSolveResult::CantPeel => {}
                    ParSolveResult::Ok(data) => break data,
                }
            }

            seed += 1;
        };

        pl.info(format_args!(
            "bits/keys: {}",
            data.len() as f64 * bit_width as f64 / num_keys as f64
        ));

        Ok(VFunc {
            seed,
            log2_l,
            high_bits: chunk_high_bits,
            chunk_mask,
            num_keys,
            segment_size,
            values: data.into(),
            _marker_t: std::marker::PhantomData,
            _marker_o: std::marker::PhantomData,
        })
    }
}
