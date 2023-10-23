/*
*
* SPDX-FileCopyrightText: 2023 Sebastiano Vigna
*
* SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
*/

use crate::bits::*;
use crate::prelude::*;
use crate::traits::bit_field_slice::{self, Word};
use common_traits::{AsBytes, AtomicUnsignedInt, IntoAtomic};
use dsi_progress_logger::ProgressLogger;
use epserde::prelude::*;
use log::warn;
use log::*;
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::thread;
use Ordering::Relaxed;

const PARAMS: [(usize, usize, f64); 15] = [
    (0, 1, 1.23),
    (10000, 32, 1.23),
    (12500, 32, 1.22),
    (19531, 32, 1.21),
    (30516, 32, 1.20),
    (38145, 64, 1.19),
    (47681, 64, 1.18),
    (74501, 64, 1.17),
    (116407, 64, 1.16),
    (227356, 64, 1.15),
    (355243, 128, 1.14),
    (693832, 128, 1.13),
    (2117406, 128, 1.12),
    (6461807, 256, 1.11),
    (60180252, 512, 1.10),
];

#[derive(Debug, Default)]
struct EdgeList(usize);
impl EdgeList {
    const DEG_SHIFT: usize = usize::BITS as usize - 16;
    const EDGE_INDEX_MASK: usize = (1_usize << EdgeList::DEG_SHIFT) - 1;
    const DEG: usize = 1_usize << EdgeList::DEG_SHIFT;
    const MAX_DEG: usize = usize::MAX >> EdgeList::DEG_SHIFT;

    #[inline(always)]
    fn add(&mut self, edge: usize) {
        debug_assert!(self.degree() <= Self::MAX_DEG);
        self.0 += EdgeList::DEG;
        debug_assert!(self.0 + edge < Self::DEG);
        self.0 += edge;
    }

    #[inline(always)]
    fn remove(&mut self, edge: usize) {
        debug_assert!(self.degree() > 0);
        self.0 -= EdgeList::DEG;
        self.0 -= edge;
    }

    #[inline(always)]
    fn degree(&self) -> usize {
        self.0 >> EdgeList::DEG_SHIFT
    }

    #[inline(always)]
    fn edge_index(&self) -> usize {
        self.0 & EdgeList::EDGE_INDEX_MASK
    }

    #[inline(always)]
    fn dec(&mut self) {
        debug_assert!(self.degree() > 0);
        self.0 -= EdgeList::DEG;
    }
}

/**

Static functions with 10%-11% space overhead for large key sets,
fast parallel construction, and fast queries.

Keys must implement the [`ToSig`] trait, which provides a method to
compute a 128-bit signature of the key.

The output type `O` can be selected to be any of the unsigned integer types
with an atomic counterpart; The default is `usize`.

*/

#[derive(Epserde, Debug, Default)]
pub struct VFunc<
    T: ToSig,
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

/**

Chunk and edge information is derived from the 128-bit signatures of the keys.
More precisely, each signature is made of two 64-bit integers `h` and `l`, and then:

- the `high_bits` most significant bits of `h` are used to select a chunk;
- the `log2_l` least significant bits of the upper 32 bits of `h` are used to select a segment;
- the lower 32 bits of `h` are used to select the virst vertex;
- the upper 32 bits of `l` are used to select the second vertex;
- the lower 32 bits of `l` are used to select the third vertex.

*/
impl<
        T: ToSig,
        O: ZeroCopy + SerializeInner + DeserializeInner + Word + IntoAtomic,
        S: bit_field_slice::BitFieldSlice<O>,
    > VFunc<T, O, S>
where
    O::AtomicType: AtomicUnsignedInt + AsBytes,
    BitFieldVec<O>: From<AtomicBitFieldVec<O, Vec<O::AtomicType>>>,
{
    #[inline(always)]
    #[must_use]
    fn chunk(sig: &[u64; 2], high_bits: u32, chunk_mask: u32) -> usize {
        (sig[0].rotate_left(high_bits) & chunk_mask as u64) as usize
    }

    #[inline(always)]
    #[must_use]
    fn edge(sig: &[u64; 2], log2_l: u32, segment_size: usize) -> [usize; 3] {
        let first_segment = (sig[0] >> 32 & ((1 << log2_l) - 1)) as usize;
        let start = first_segment * segment_size;
        [
            (((sig[0] & 0xFFFFFFFF) * segment_size as u64) >> 32) as usize + start,
            (((sig[1] >> 32) * segment_size as u64) >> 32) as usize + start + segment_size,
            (((sig[1] & 0xFFFFFFFF) * segment_size as u64) >> 32) as usize
                + start
                + 2 * segment_size,
        ]
    }

    pub fn get_by_sig(&self, sig: &[u64; 2]) -> O {
        let edge = Self::edge(sig, self.log2_l, self.segment_size);
        let chunk = Self::chunk(sig, self.high_bits, self.chunk_mask);
        // chunk * self.segment_size * (2^log2_l + 2)
        let chunk_offset = chunk * ((self.segment_size << self.log2_l) + (self.segment_size << 1));
        self.values.get(edge[0] + chunk_offset)
            ^ self.values.get(edge[1] + chunk_offset)
            ^ self.values.get(edge[2] + chunk_offset)
    }

    #[inline(always)]
    pub fn get(&self, key: &T) -> O {
        self.get_by_sig(&T::to_sig(key, self.seed))
    }

    pub fn len(&self) -> usize {
        self.num_keys
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Creates a new function with given keys and values.
    pub fn new<
        I: std::iter::IntoIterator<Item = T> + Clone,
        V: std::iter::IntoIterator<Item = O> + Clone,
    >(
        keys: I,
        into_values: &V,
        pl: &mut Option<&mut ProgressLogger>,
    ) -> VFunc<T, O> {
        use crate::traits::bit_field_slice::AtomicBitFieldSlice;
        // Loop until success or duplicate detection
        let mut dup_count = 0;
        for seed in 0.. {
            if let Some(pl) = pl.as_mut() {
                pl.start("Reading input...")
            }

            let mut values = into_values.clone().into_iter();
            let mut max_value = O::ZERO;
            let mut sigs = keys
                .clone()
                .into_iter()
                .map(|x| {
                    let v = values.next().expect("Not enough values");
                    if let Some(pl) = pl.as_mut() {
                        pl.light_update();
                    }
                    max_value = Ord::max(max_value, v);
                    (T::to_sig(&x, seed), v)
                })
                .collect::<Vec<_>>();

            let bit_width = max_value.len() as usize;
            info!("max value = {}, bit width = {}", max_value, bit_width);

            if let Some(pl) = pl.as_mut() {
                pl.done_with_count(sigs.len());
            }

            /*if values.next().is_some() {
                // TODO
                panic!("Too many values");
            }*/

            let num_keys = sigs.len();
            let (chunk_high_bits, max_num_threads, l, c);

            if num_keys < PARAMS[PARAMS.len() - 2].0 {
                // Too few keys, we cannot split into chunks
                chunk_high_bits = 0;
                max_num_threads = 1;
                (_, l, c) = PARAMS
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
                (_, l, c) = PARAMS[PARAMS.len() - 2];
            } else {
                // We are in the 1.10 regime. We can increase the number
                // of threads, but we need to be careful not to use too
                // much memory.

                let eps = 0.001;
                chunk_high_bits = {
                    let t = (sigs.len() as f64 * eps * eps / 2.0).ln();

                    if t > 0.0 {
                        ((t - t.ln()) / 2_f64.ln()).ceil()
                    } else {
                        0.0
                    }
                }
                .min(10.0) // More than 1000 chunks make no sense
                .min((num_keys / PARAMS[PARAMS.len() - 1].0).ilog2() as f64) // Let's keep the chunks in the 1.10 regime
                    as u32;
                max_num_threads = 1.max((1 << chunk_high_bits) / 8);
                (_, l, c) = PARAMS[PARAMS.len() - 1];
            }

            info!("high bits = {}, l = {}, c = {}", chunk_high_bits, l, c);

            let log2_l = l.ilog2();
            let num_chunks = 1 << chunk_high_bits;
            let chunk_mask = (1u32 << chunk_high_bits) - 1;

            if let Some(pl) = pl.as_mut() {
                pl.start("Sorting...")
            }
            sigs.par_sort_unstable();
            if let Some(pl) = pl.as_mut() {
                pl.done_with_count(num_keys);
            }

            if let Some(pl) = pl.as_mut() {
                pl.start("Checking for duplicates...")
            }

            let mut counts = vec![0; num_chunks];
            let mut dup = false;

            counts[Self::chunk(&sigs[0].0, chunk_high_bits, chunk_mask)] += 1;

            for w in sigs.windows(2) {
                counts[Self::chunk(&w[1].0, chunk_high_bits, chunk_mask)] += 1;
                if w[0].0 == w[1].0 {
                    dup = true;
                    break;
                }
            }

            if let Some(pl) = pl.as_mut() {
                pl.done_with_count(num_keys);
            }

            if dup {
                if dup_count >= 3 {
                    panic!(
                        "Duplicate keys (duplicate 128-bit signatures with four different seeds)"
                    );
                }
                warn!("Duplicate 128-bit signature, trying again...");
                dup_count += 1;
                continue;
            }

            let mut cumul = vec![0; num_chunks + 1];
            for i in 0..num_chunks {
                cumul[i + 1] = cumul[i] + counts[i];
            }

            let segment_size =
                ((*counts.iter().max().unwrap() as f64 * c).ceil() as usize + l + 1) / (l + 2);
            let num_vertices = segment_size * (l + 2);
            info!(
                "Size {:.2}%",
                (100.0 * (num_vertices * num_chunks) as f64) / (sigs.len() as f64 * c)
            );

            let data = AtomicBitFieldVec::<O>::new(bit_width, num_vertices * num_chunks);

            let chunk = AtomicUsize::new(0);
            let fail = AtomicBool::new(false);
            let num_threads = num_chunks.min(num_cpus::get()).min(max_num_threads);
            info!("Using {} threads", num_threads);

            thread::scope(|s| {
                for _ in 0..num_threads {
                    s.spawn(|| loop {
                        let chunk = chunk.fetch_add(1, Relaxed);
                        if chunk >= num_chunks {
                            break;
                        }
                        let sigs = &sigs[cumul[chunk]..cumul[chunk + 1]];

                        let mut pl = ProgressLogger::default();
                        pl.expected_updates = Some(sigs.len());

                        pl.start(format!(
                            "Generating graph for chunk {}/{}...",
                            chunk, num_chunks
                        ));
                        let mut edge_lists = Vec::new();
                        edge_lists.resize_with(num_vertices, EdgeList::default);

                        sigs.iter().enumerate().for_each(|(edge_index, sig)| {
                            for &v in Self::edge(&sig.0, log2_l, segment_size).iter() {
                                edge_lists[v].add(edge_index);
                            }
                        });
                        pl.done_with_count(sigs.len());

                        let next = AtomicUsize::new(0);
                        let incr = 1024;

                        pl.start(format!(
                            "Peeling graph for chunk {}/{}...",
                            chunk, num_chunks
                        ));

                        let mut stack = Vec::new();
                        loop {
                            if fail.load(Ordering::Relaxed) {
                                return;
                            }
                            let start = next.fetch_add(incr, Ordering::Relaxed);
                            if start >= num_vertices {
                                break;
                            }
                            for v in start..(start + incr).min(num_vertices) {
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

                                    edge_lists[v].dec();
                                    let edge_index = edge_lists[v].edge_index();

                                    stack[curr] = v;
                                    curr += 1;
                                    // Degree is necessarily 0
                                    for &x in
                                        Self::edge(&sigs[edge_index].0, log2_l, segment_size).iter()
                                    {
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
                        }
                        if sigs.len() != stack.len() {
                            fail.store(true, Ordering::Relaxed);
                        }
                        pl.done_with_count(sigs.len());

                        pl.start(format!(
                            "Assigning values for chunk {}/{}...",
                            chunk, num_chunks
                        ));
                        while let Some(mut v) = stack.pop() {
                            let edge_index = edge_lists[v].edge_index();
                            let mut edge = Self::edge(&sigs[edge_index].0, log2_l, segment_size);
                            let chunk_offset = chunk * num_vertices;
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
                            // TODO
                            data.set(v, sigs[edge_index].1 ^ value, Relaxed);

                            debug_assert_eq!(
                                data.get(edge[0], Relaxed)
                                    ^ data.get(edge[1], Relaxed)
                                    ^ data.get(edge[2], Relaxed),
                                sigs[edge_index].1
                            );
                        }
                        pl.done_with_count(sigs.len());
                        pl.start(format!("Completed chunk {}/{}.", chunk, num_chunks));
                    });
                }
            });

            if fail.load(Ordering::Relaxed) {
                warn!("Failed peeling, trying again...")
            } else {
                info!(
                    "bits/keys: {}",
                    data.len() as f64 * bit_width as f64 / sigs.len() as f64,
                );
                return VFunc {
                    seed,
                    log2_l,
                    high_bits: chunk_high_bits,
                    chunk_mask,
                    num_keys: sigs.len(),
                    segment_size,
                    values: data.into(),
                    _marker_t: std::marker::PhantomData,
                    _marker_o: std::marker::PhantomData,
                };
            }
        }
        unreachable!("There are infinite possible seeds.")
    }

    pub fn new_offline<
        I: std::iter::IntoIterator<Item = T> + Clone,
        V: std::iter::IntoIterator<Item = O> + Clone,
    >(
        keys: I,
        into_values: &V,
        pl: &mut Option<&mut ProgressLogger>,
    ) -> anyhow::Result<VFunc<T, O>> {
        use crate::traits::bit_field_slice::AtomicBitFieldSlice;
        // Loop until success or duplicate detection
        let mut dup_count = 0;
        for seed in 0.. {
            if let Some(pl) = pl.as_mut() {
                pl.start("Reading input...")
            }

            let store_bits = 12;
            let mut sig_sorter = SigStore::<O>::new(8, store_bits).unwrap();
            let mut values = into_values.clone().into_iter();
            let mut max_value = O::ZERO;
            sig_sorter.extend(keys.clone().into_iter().map(|x| {
                if let Some(pl) = pl.as_mut() {
                    pl.light_update();
                }
                let v = values.next().expect("Not enough values");
                max_value = Ord::max(max_value, v);
                (T::to_sig(&x, seed), v)
            }))?;

            let num_keys = sig_sorter.num_keys();

            let bit_width = max_value.len() as usize;
            info!("max value = {}, bit width = {}", max_value, bit_width);

            if let Some(pl) = pl.as_mut() {
                pl.done_with_count(num_keys);
            }

            /*if values.next().is_some() {
                // TODO
                panic!("Too many values");
            }*/

            let (chunk_high_bits, max_num_threads, l, c);

            if num_keys < PARAMS[PARAMS.len() - 2].0 {
                // Too few keys, we cannot split into chunks
                chunk_high_bits = 0;
                max_num_threads = 1;
                (_, l, c) = PARAMS
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
                (_, l, c) = PARAMS[PARAMS.len() - 2];
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
                max_num_threads = 1.max((1 << chunk_high_bits) / 8);
                (_, l, c) = PARAMS[PARAMS.len() - 1];
            }

            info!(
                "chunk high bits = {}, l = {}, c = {}",
                chunk_high_bits, l, c
            );

            let log2_l = l.ilog2();
            let num_chunks = 1 << chunk_high_bits;
            let chunk_mask = (1u32 << chunk_high_bits) - 1;

            let (chunk_store, chunk_sizes);
            if let Ok(t) = sig_sorter.into_chunk_store(chunk_high_bits) {
                (chunk_store, chunk_sizes) = t;
            } else {
                if dup_count >= 3 {
                    panic!(
                        "Duplicate keys (duplicate 128-bit signatures with four different seeds)"
                    );
                }
                warn!("Duplicate 128-bit signature, trying again...");
                dup_count += 1;
                continue;
            }

            let mut cumul = vec![0; num_chunks + 1];
            for i in 0..num_chunks {
                cumul[i + 1] = cumul[i] + chunk_sizes[i];
            }

            let segment_size =
                ((*chunk_sizes.iter().max().unwrap() as f64 * c).ceil() as usize + l + 1) / (l + 2);
            let num_vertices = segment_size * (l + 2);
            info!(
                "Size {:.2}%",
                (100.0 * (num_vertices * num_chunks) as f64) / (num_keys as f64 * c)
            );

            let data = AtomicBitFieldVec::<O>::new(bit_width, num_vertices * num_chunks);

            let fail = AtomicBool::new(false);
            let mutex = std::sync::Arc::new(Mutex::new(chunk_store));
            let num_threads = num_chunks.min(num_cpus::get()).min(max_num_threads);
            info!("Using {} threads", num_threads);

            thread::scope(|s| {
                for _ in 0..num_threads {
                    s.spawn(|| loop {
                        let chunks;
                        {
                            let next = mutex.lock().unwrap().next();
                            if next.is_none() {
                                return;
                            }
                            chunks = next.unwrap();
                        }
                        for (chunk_index, chunk) in chunks {
                            let mut pl = ProgressLogger::default();
                            pl.expected_updates = Some(num_keys);

                            pl.start(format!(
                                "Generating graph for chunk {}/{}...",
                                chunk_index, num_chunks
                            ));
                            let mut edge_lists = Vec::new();
                            edge_lists.resize_with(num_vertices, EdgeList::default);

                            chunk.iter().enumerate().for_each(|(edge_index, sig)| {
                                for &v in Self::edge(&sig.0, log2_l, segment_size).iter() {
                                    edge_lists[v].add(edge_index);
                                }
                            });
                            pl.done_with_count(chunk.len());

                            let next = AtomicUsize::new(0);
                            let incr = 1024;

                            pl.start(format!(
                                "Peeling graph for chunk {}/{}...",
                                chunk_index, num_chunks
                            ));

                            let mut stack = Vec::new();
                            loop {
                                if fail.load(Ordering::Relaxed) {
                                    return;
                                }
                                let start = next.fetch_add(incr, Ordering::Relaxed);
                                if start >= num_vertices {
                                    break;
                                }
                                for v in start..(start + incr).min(num_vertices) {
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

                                        edge_lists[v].dec();
                                        let edge_index = edge_lists[v].edge_index();

                                        stack[curr] = v;
                                        curr += 1;
                                        // Degree is necessarily 0
                                        for &x in
                                            Self::edge(&chunk[edge_index].0, log2_l, segment_size)
                                                .iter()
                                        {
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
                            }
                            if chunk.len() != stack.len() {
                                fail.store(true, Ordering::Relaxed);
                                return;
                            }

                            pl.done_with_count(chunk.len());

                            pl.start(format!(
                                "Assigning values for chunk {}/{}...",
                                chunk_index, num_chunks
                            ));
                            while let Some(mut v) = stack.pop() {
                                let edge_index = edge_lists[v].edge_index();
                                let mut edge =
                                    Self::edge(&chunk[edge_index].0, log2_l, segment_size);
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
                                // TODO
                                data.set(v, chunk[edge_index].1 ^ value, Relaxed);

                                debug_assert_eq!(
                                    data.get(edge[0], Relaxed)
                                        ^ data.get(edge[1], Relaxed)
                                        ^ data.get(edge[2], Relaxed),
                                    chunk[edge_index].1
                                );
                            }
                            pl.done_with_count(chunk.len());
                            pl.start(format!("Completed chunk {}/{}.", chunk_index, num_chunks));
                        }
                    });
                }
            });

            if fail.load(Ordering::Relaxed) {
                warn!("Failed peeling, trying again...")
            } else {
                info!(
                    "bits/keys: {}",
                    data.len() as f64 * bit_width as f64 / num_keys as f64,
                );
                return Ok(VFunc {
                    seed,
                    log2_l,
                    high_bits: chunk_high_bits,
                    chunk_mask,
                    num_keys,
                    segment_size,
                    values: data.into(),
                    _marker_t: std::marker::PhantomData,
                    _marker_o: std::marker::PhantomData,
                });
            }
        }
        unreachable!("There are infinite possible seeds.")
    }
}
