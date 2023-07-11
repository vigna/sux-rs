use anyhow::Result;
use clap::Parser;
use dsi_progress_logger::ProgressLogger;
use rayon::prelude::*;
use std::collections::HashSet;
use std::io::{BufRead, BufReader};
use std::mem::{self, size_of};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::thread;
use sux::spooky::spooky_short;
use sux::spooky::spooky_short_mix;
use sux::spooky::SC_CONST;
use Ordering::Relaxed;

#[derive(Parser, Debug)]
#[command(about = "Functions", long_about = None)]
struct Args {
    filename: String,
    threads: usize,
}

pub trait Remap {
    fn remap(key: &Self, seed: u64) -> [u64; 2];
}

impl<T: AsRef<[u8]>> Remap for T {
    fn remap(key: &Self, seed: u64) -> [u64; 2] {
        let spooky = spooky_short(key.as_ref(), seed);
        [spooky[0], spooky[1]]
    }
}

#[derive(Debug, Default)]

struct EdgeList(usize);
impl EdgeList {
    const DEG_SHIFT: usize = 8 * size_of::<usize>() - 10;
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

pub struct Function<T: Remap> {
    seed: u64,
    num_keys: usize,
    high_bits: usize,
    segment_size: usize,
    values: Box<[u64]>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Remap> Function<T> {
    #[inline(always)]
    #[must_use]
    fn chunk(sig: &[u64; 2], high_bits: usize) -> usize {
        (sig[0] >> 63 - high_bits >> 1) as usize
    }

    #[inline(always)]
    #[must_use]
    fn edge(sig: &[u64; 2], segment_size: usize) -> [usize; 3] {
        let tuple = spooky_short_rehash(sig, 0);
        let first_segment = tuple[3] as usize % 128;
        [
            ((tuple[0] as u128) * (segment_size as u128) >> 64) as usize
                + (first_segment + 0) * segment_size,
            ((tuple[1] as u128) * (segment_size as u128) >> 64) as usize
                + (first_segment + 1) * segment_size,
            ((tuple[2] as u128) * (segment_size as u128) >> 64) as usize
                + (first_segment + 2) * segment_size,
        ]
    }

    pub fn get_by_sig(&self, sig: &[u64; 2]) -> u64 {
        let edge = Self::edge(&sig, self.segment_size);
        let chunk = Self::chunk(sig, self.high_bits);
        let chunk_offset = chunk * self.segment_size * 130;
        self.values[edge[0] + chunk_offset]
            ^ self.values[edge[1] + chunk_offset]
            ^ self.values[edge[2] + chunk_offset]
    }

    #[inline(always)]
    pub fn get(&self, key: &T) -> u64 {
        self.get_by_sig(&T::remap(key, self.seed))
    }

    pub fn len(&self) -> usize {
        self.num_keys
    }

    pub fn new<I: std::iter::Iterator<Item = T>, V: std::iter::Iterator<Item = u64>>(
        keys: I,
        values: &mut V,
        bit_width: usize,
        threads: usize,
        pl: &mut Option<&mut ProgressLogger>,
    ) -> Function<T> {
        pl.as_mut().map(|pl| pl.start("Reading input..."));
        let mut sigs = keys
            .map(|x| (T::remap(&x, 0), values.next().unwrap()))
            .collect::<Vec<_>>();
        pl.as_mut().map(|pl| pl.done());

        let high_bits = 0;
        let num_chunks = 1 << high_bits;
        let (counts, cumul) = count_sort(&mut sigs, num_chunks, |x| Self::chunk(&x.0, high_bits));

        let segment_size =
            ((*counts.iter().max().unwrap() as f64 * 1.12).ceil() as usize + 129) / 130;
        let num_vertices = segment_size * 130;

        let mut values = Vec::new();
        values.resize_with(num_vertices * num_chunks, || AtomicU64::new(0));

        let num_peeled = AtomicUsize::new(0);
        let chunk = AtomicUsize::new(0);

        thread::scope(|s| {
            for _ in 0..num_chunks {
                s.spawn(|| {
                    let chunk = chunk.fetch_add(1, Relaxed);
                    let sigs = &sigs[cumul[chunk]..cumul[chunk + 1]];

                    let mut pl = ProgressLogger::default();
                    pl.expected_updates = Some(sigs.len());

                    pl.start("Generating graph...");
                    let mut edge_lists = Vec::new();
                    edge_lists.resize_with(num_vertices, || EdgeList::default());

                    sigs.iter().enumerate().for_each(|(edge_index, sig)| {
                        for &v in Self::edge(&sig.0, segment_size).iter() {
                            edge_lists[v].add(edge_index);
                        }
                    });
                    pl.done_with_count(sigs.len());

                    let mut peeled = Vec::new();
                    peeled.resize_with(sigs.len(), || AtomicBool::new(false));

                    let next = AtomicUsize::new(0);
                    let incr = 1024;

                    pl.start("Peeling...");
                    let stacks = Mutex::new(vec![]);

                    let mut stack = Vec::new();
                    loop {
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

                                edge_lists[v].dec();
                                if edge_lists[v].degree() != 0 {
                                    debug_assert_eq!(edge_lists[v].degree(), 0, "v = {}", v);
                                    continue; // Skip no longer useful entries
                                }
                                let edge_index = edge_lists[v].edge_index();
                                if peeled[edge_index].swap(true, Ordering::Relaxed) {
                                    // Someone already peeled this edge
                                    continue;
                                }

                                stack[curr] = v;
                                curr += 1;
                                // Degree is necessarily 0
                                for &x in Self::edge(&sigs[edge_index].0, segment_size).iter() {
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
                    num_peeled.fetch_add(stack.len(), Relaxed);
                    stacks.lock().unwrap().push(stack);

                    pl.done_with_count(sigs.len());

                    pl.start("Assigning...");
                    let mut t = HashSet::new();
                    let mut stacks = stacks.lock().unwrap();
                    while let Some(mut stack) = stacks.pop() {
                        let mut s = HashSet::new();
                        while let Some(mut v) = stack.pop() {
                            let edge_index = edge_lists[v].edge_index();
                            let mut edge = Self::edge(&sigs[edge_index].0, segment_size);
                            let chunk_offset = chunk * num_vertices;
                            v += chunk_offset;
                            edge.iter_mut().for_each(|v| {
                                *v += chunk_offset;
                            });
                            assert!(s.insert(v));
                            assert!(t.insert(v));
                            let value = if v == edge[0] {
                                values[edge[1]].load(Relaxed) ^ values[edge[2]].load(Relaxed)
                            } else if v == edge[1] {
                                values[edge[0]].load(Relaxed) ^ values[edge[2]].load(Relaxed)
                            } else {
                                debug_assert_eq!(v, edge[2]);
                                values[edge[0]].load(Relaxed) ^ values[edge[1]].load(Relaxed)
                            };
                            values[v].store(&sigs[edge_index].1 ^ value, Relaxed);

                            assert_eq!(
                                values[edge[0]].load(Relaxed)
                                    ^ values[edge[1]].load(Relaxed)
                                    ^ values[edge[2]].load(Relaxed),
                                sigs[edge_index].1
                            );
                        }
                    }
                    pl.done_with_count(sigs.len());
                });
            }
        });
        assert_eq!(sigs.len(), num_peeled.load(Relaxed));
        println!(
            "bits/keys: {}",
            values.len() as f64 * 64.0 / sigs.len() as f64,
        );
        Function {
            seed: 0,
            num_keys: sigs.len(),
            high_bits,
            segment_size,
            values: unsafe {
                std::mem::transmute::<Vec<AtomicU64>, Vec<u64>>(values).into_boxed_slice()
            },
            _phantom: std::marker::PhantomData,
        }
    }
}

fn count_sort<T: Copy + Clone, F: Fn(&T) -> usize>(
    data: &mut [T],
    num_keys: usize,
    key: F,
) -> (Vec<usize>, Vec<usize>) {
    if num_keys == 1 {
        return (vec![data.len()], vec![0, data.len()]);
    }
    let mut counts = Vec::new();
    counts.resize(num_keys, 0);

    for sig in &*data {
        counts[key(sig)] += 1;
    }

    let mut cumul = Vec::new();
    cumul.resize(counts.len() + 1, 0);
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

#[inline(always)]
#[must_use]
pub const fn spooky_short_rehash(signature: &[u64; 2], seed: u64) -> [u64; 4] {
    spooky_short_mix([
        seed,
        SC_CONST.wrapping_add(signature[0]),
        SC_CONST.wrapping_add(signature[1]),
        SC_CONST,
    ])
}

fn main() -> Result<()> {
    stderrlog::new()
        .verbosity(2)
        .timestamp(stderrlog::Timestamp::Second)
        .init()
        .unwrap();

    let args = Args::parse();

    let mut pl = ProgressLogger::default();

    let file = std::fs::File::open(&args.filename)?;
    let func = Function::new(
        BufReader::new(file).lines().map(|line| line.unwrap()),
        &mut (0..).into_iter(),
        64,
        args.threads,
        &mut Some(&mut pl),
    );

    let file = std::fs::File::open(&args.filename)?;
    let keys = BufReader::new(file)
        .lines()
        .map(|line| line.unwrap())
        .take(10_000_000)
        .collect::<Vec<_>>();

    pl.start("Querying...");
    for (index, key) in keys.iter().enumerate() {
        assert_eq!(index, func.get(key) as usize);
    }
    pl.done_with_count(keys.len());

    Ok(())
}
