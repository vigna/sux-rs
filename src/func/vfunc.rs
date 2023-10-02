/*
*
* SPDX-FileCopyrightText: 2023 Sebastiano Vigna
*
* SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
*/

/*!

No-nonsense static functions with 10%-11% space overhead,
fast parallel construction, and fast queries.

*/

use crate::prelude::{
    spooky::*, BitFieldSlice, BitFieldSliceAtomic, BitFieldSliceCore, CompactArray, SigStore,
};
use crate::traits::convert_to::ConvertTo;
use crate::BitOps;
use dsi_progress_logger::ProgressLogger;
use epserde::Epserde;
use log::warn;
use log::*;
use rayon::prelude::*;
use std::mem::{self};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::thread;
use Ordering::Relaxed;

/// This trait has to be implemented by keys. It must turn a key
/// into a random-looking 128-bit signature.
///
/// We provide implementations for all primitive types and strings
/// by turning them into slice of bytes and then hashing them with
/// [crate::mph::spooky::spooky_short].
pub trait ToSig {
    fn to_sig(key: &Self, seed: u64) -> [u64; 2];
}

impl ToSig for String {
    fn to_sig(key: &Self, seed: u64) -> [u64; 2] {
        let spooky = spooky_short(key.as_ref(), seed);
        [spooky[0], spooky[1]]
    }
}

impl ToSig for &String {
    fn to_sig(key: &Self, seed: u64) -> [u64; 2] {
        let spooky = spooky_short(key.as_ref(), seed);
        [spooky[0], spooky[1]]
    }
}

impl ToSig for str {
    fn to_sig(key: &Self, seed: u64) -> [u64; 2] {
        let spooky = spooky_short(key.as_ref(), seed);
        [spooky[0], spooky[1]]
    }
}

impl ToSig for &str {
    fn to_sig(key: &Self, seed: u64) -> [u64; 2] {
        let spooky = spooky_short(key.as_ref(), seed);
        [spooky[0], spooky[1]]
    }
}

macro_rules! to_sig_prim {
    ($($ty:ty),*) => {$(
        impl ToSig for $ty {
            fn to_sig(key: &Self, seed: u64) -> [u64; 2] {
                let spooky = spooky_short(&key.to_ne_bytes(), seed);
                [spooky[0], spooky[1]]
            }
        }
    )*};
}

to_sig_prim!(isize, usize, i8, i16, i32, i64, i128, u8, u16, u32, u64, u128);

macro_rules! to_sig_slice {
    ($($ty:ty),*) => {$(
        impl ToSig for &[$ty] {
            fn to_sig(key: &Self, seed: u64) -> [u64; 2] {
                // Alignemnt to u8 never fails or leave trailing/leading bytes
                let spooky = spooky_short(unsafe {key.align_to::<u8>().1 }, seed);
                [spooky[0], spooky[1]]
            }
        }
    )*};
}

to_sig_slice!(isize, usize, i8, i16, i32, i64, i128, u8, u16, u32, u64, u128);

const PARAMS: [(usize, usize, f64); 15] = [
    (0, 1, 1.23),
    (13, 32, 1.22),
    (14, 32, 1.21),
    (15, 32, 1.20),
    (16, 64, 1.18),
    (17, 64, 1.16),
    (18, 64, 1.15),
    (19, 128, 1.14),
    (20, 128, 1.13),
    (21, 256, 1.12),
    (22, 512, 1.12),
    (23, 256, 1.11),
    (24, 512, 1.11),
    (25, 256, 1.11),
    (26, 512, 1.10),
];

#[derive(Debug, Default)]
struct EdgeList(usize);
impl EdgeList {
    const DEG_SHIFT: usize = usize::BITS as usize - 10;
    const EDGE_INDEX_MASK: usize = (1_usize << EdgeList::DEG_SHIFT) - 1;
    const DEG: usize = 1_usize << EdgeList::DEG_SHIFT;

    #[inline(always)]
    fn add(&mut self, edge: usize) {
        self.0 += EdgeList::DEG | edge;
    }

    #[inline(always)]
    fn remove(&mut self, edge: usize) {
        self.0 -= EdgeList::DEG | edge;
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
        self.0 -= EdgeList::DEG;
    }
}

/**

A static function from key implementing [ToSig] to arbitrary values.

*/
#[derive(Epserde, Debug, Default)]
pub struct VFunc<T: ToSig, S: BitFieldSlice = CompactArray<Vec<usize>>> {
    seed: u64,
    log2_l: u32,
    high_bits: u32,
    chunk_mask: u32,
    num_keys: usize,
    segment_size: usize,
    values: S,
    _phantom: std::marker::PhantomData<T>,
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
impl<T: ToSig, S: BitFieldSlice> VFunc<T, S> {
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

    pub fn get_by_sig(&self, sig: &[u64; 2]) -> usize {
        let edge = Self::edge(sig, self.log2_l, self.segment_size);
        let chunk = Self::chunk(sig, self.high_bits, self.chunk_mask);
        // chunk * self.segment_size * (2^log2_l + 2)
        let chunk_offset = chunk * ((self.segment_size << self.log2_l) + (self.segment_size << 1));
        self.values.get(edge[0] + chunk_offset)
            ^ self.values.get(edge[1] + chunk_offset)
            ^ self.values.get(edge[2] + chunk_offset)
    }

    #[inline(always)]
    pub fn get(&self, key: &T) -> usize {
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
        V: std::iter::IntoIterator<Item = usize> + Clone,
    >(
        keys: I,
        into_values: &V,
        pl: &mut Option<&mut ProgressLogger>,
    ) -> VFunc<T> {
        // Loop until success or duplicate detection
        let mut dup_count = 0;
        for seed in 0.. {
            if let Some(pl) = pl.as_mut() {
                pl.start("Reading input...")
            }

            let mut values = into_values.clone().into_iter();
            let mut max_value = 0;
            let mut sigs = keys
                .clone()
                .into_iter()
                .map(|x| {
                    let v = values.next().expect("Not enough values");
                    if let Some(pl) = pl.as_mut() {
                        pl.light_update();
                    }
                    max_value = max_value.max(v);
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

            let eps = 0.001;
            let mut chunk_high_bits = {
                let t = (sigs.len() as f64 * eps * eps / 2.0).ln();

                if t > 0.0 {
                    ((t - t.ln()) / 2_f64.ln()).ceil()
                } else {
                    0.0
                }
            }
            .min(10.0) as u32; // More than 1000 chunks make no sense

            let (l, c) = {
                loop {
                    let (_, l, c) = PARAMS
                        .iter()
                        .rev()
                        .filter(|(log_n, _l, _c)| (1 << log_n) <= (num_keys >> chunk_high_bits))
                        .copied()
                        .next()
                        .unwrap(); // first log_n is 0, so this is always valid
                    if chunk_high_bits == 0 || c <= 1.11 {
                        break (l, c);
                    }
                    chunk_high_bits -= 1;
                }
            };

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

            let data = CompactArray::new_atomic(bit_width, num_vertices * num_chunks);

            let chunk = AtomicUsize::new(0);
            let fail = AtomicBool::new(false);

            thread::scope(|s| {
                for _ in 0..num_chunks.min(num_cpus::get()) {
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
                        let stacks = Mutex::new(vec![]);

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
                        stacks.lock().unwrap().push(stack);

                        pl.done_with_count(sigs.len());

                        pl.start(format!(
                            "Assigning values for chunk {}/{}...",
                            chunk, num_chunks
                        ));
                        let mut stacks = stacks.lock().unwrap();
                        while let Some(mut stack) = stacks.pop() {
                            while let Some(mut v) = stack.pop() {
                                let edge_index = edge_lists[v].edge_index();
                                let mut edge =
                                    Self::edge(&sigs[edge_index].0, log2_l, segment_size);
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

                                assert_eq!(
                                    data.get(edge[0], Relaxed)
                                        ^ data.get(edge[1], Relaxed)
                                        ^ data.get(edge[2], Relaxed),
                                    sigs[edge_index].1
                                );
                            }
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
                    values: data.convert_to().unwrap(),
                    _phantom: std::marker::PhantomData,
                };
            }
        }
        unreachable!("There are infinite possible seeds.")
    }

    pub fn new_offline<
        I: std::iter::IntoIterator<Item = T> + Clone,
        V: std::iter::IntoIterator<Item = usize> + Clone,
    >(
        keys: I,
        into_values: &V,
        pl: &mut Option<&mut ProgressLogger>,
    ) -> anyhow::Result<VFunc<T>> {
        // Loop until success or duplicate detection
        let mut dup_count = 0;
        for seed in 0.. {
            if let Some(pl) = pl.as_mut() {
                pl.start("Reading input...")
            }

            let mut sig_sorter = SigStore::new(8).unwrap();
            let mut values = into_values.clone().into_iter();
            let mut max_value = 0;
            sig_sorter.extend(keys.clone().into_iter().map(|x| {
                if let Some(pl) = pl.as_mut() {
                    pl.light_update();
                }
                let v = values.next().expect("Not enough values");
                max_value = max_value.max(v);
                (T::to_sig(&x, seed), v as u64)
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

            let num_keys = num_keys;

            let eps = 0.001;
            let mut chunk_high_bits = {
                let t = (num_keys as f64 * eps * eps / 2.0).ln();

                if t > 0.0 {
                    ((t - t.ln()) / 2_f64.ln()).ceil()
                } else {
                    0.0
                }
            }
            .min(10.0) as u32; // More than 1000 chunks make no sense

            let (l, c) = {
                loop {
                    let (_, l, c) = PARAMS
                        .iter()
                        .rev()
                        .filter(|(log_n, _l, _c)| (1 << log_n) <= (num_keys >> chunk_high_bits))
                        .copied()
                        .next()
                        .unwrap(); // first log_n is 0, so this is always valid
                    if chunk_high_bits == 0 || c <= 1.11 {
                        break (l, c);
                    }
                    chunk_high_bits -= 1;
                }
            };

            info!(
                "chunk high bits = {}, l = {}, c = {}",
                chunk_high_bits, l, c
            );

            let log2_l = l.ilog2();
            let num_chunks = 1 << chunk_high_bits;
            let chunk_mask = (1u32 << chunk_high_bits) - 1;

            if let Some(pl) = pl.as_mut() {
                pl.start("Sorting...")
            }
            let sorted_sig;
            if let Ok(t) = sig_sorter.into_iter(chunk_high_bits) {
                sorted_sig = t;
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

            if let Some(pl) = pl.as_mut() {
                pl.done_with_count(num_keys);
            }

            let mut cumul = vec![0; num_chunks + 1];
            for i in 0..num_chunks {
                cumul[i + 1] = cumul[i] + sorted_sig.counts()[i];
            }

            let segment_size =
                ((*sorted_sig.counts().iter().max().unwrap() as f64 * c).ceil() as usize + l + 1)
                    / (l + 2);
            let num_vertices = segment_size * (l + 2);
            info!(
                "Size {:.2}%",
                (100.0 * (num_vertices * num_chunks) as f64) / (num_keys as f64 * c)
            );

            let data = CompactArray::new_atomic(bit_width, num_vertices * num_chunks);

            let fail = AtomicBool::new(false);
            let mutex = std::sync::Arc::new(Mutex::new(sorted_sig));

            thread::scope(|s| {
                for _ in 0..num_chunks.min(num_cpus::get()) {
                    s.spawn(|| loop {
                        let (sigs, chunk);
                        {
                            let next = mutex.lock().unwrap().next();
                            if next.is_none() {
                                return;
                            }
                            (chunk, sigs) = next.unwrap();
                        }

                        let mut pl = ProgressLogger::default();
                        pl.expected_updates = Some(num_keys);

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
                        let stacks = Mutex::new(vec![]);

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
                        stacks.lock().unwrap().push(stack);

                        pl.done_with_count(sigs.len());

                        pl.start(format!(
                            "Assigning values for chunk {}/{}...",
                            chunk, num_chunks
                        ));
                        let mut stacks = stacks.lock().unwrap();
                        while let Some(mut stack) = stacks.pop() {
                            while let Some(mut v) = stack.pop() {
                                let edge_index = edge_lists[v].edge_index();
                                let mut edge =
                                    Self::edge(&sigs[edge_index].0, log2_l, segment_size);
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
                                data.set(v, sigs[edge_index].1 as usize ^ value, Relaxed);

                                assert_eq!(
                                    data.get(edge[0], Relaxed)
                                        ^ data.get(edge[1], Relaxed)
                                        ^ data.get(edge[2], Relaxed),
                                    sigs[edge_index].1 as usize
                                );
                            }
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
                    data.len() as f64 * bit_width as f64 / num_keys as f64,
                );
                return Ok(VFunc {
                    seed,
                    log2_l,
                    high_bits: chunk_high_bits,
                    chunk_mask,
                    num_keys,
                    segment_size,
                    values: data.convert_to().unwrap(),
                    _phantom: std::marker::PhantomData,
                });
            }
        }
        unreachable!("There are infinite possible seeds.")
    }
}

fn _count_sort<T: Copy + Clone, F: Fn(&T) -> usize>(
    data: &mut [T],
    num_keys: usize,
    key: F,
) -> (Vec<usize>, Vec<usize>) {
    if num_keys == 1 {
        return (vec![data.len()], vec![0, data.len()]);
    }
    let mut counts = vec![0; num_keys];

    for sig in &*data {
        counts[key(sig)] += 1;
    }

    let mut cumul = vec![0; counts.len() + 1];
    for i in 1..cumul.len() {
        cumul[i] += cumul[i - 1] + counts[i - 1];
    }
    let end = data.len() - counts.last().unwrap();

    let mut pos = cumul[1..].to_vec();
    let mut i = 0;
    while i < end {
        let mut sig = data[i];

        loop {
            let slot = key(&sig);
            pos[slot] -= 1;
            if pos[slot] <= i {
                break;
            }
            mem::swap(&mut data[pos[slot]], &mut sig);
        }
        data[i] = sig;
        i += counts[key(&sig)];
    }

    (counts, cumul)
}
