use anyhow::Result;
use clap::Parser;
use dsi_progress_logger::ProgressLogger;
use rayon::prelude::*;
use std::io::{BufRead, BufReader};
use std::mem::{self, size_of};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::thread;
use sux::prelude::CompactArray;
use sux::prelude::VSlice;
use sux::prelude::VSliceMut;
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
    fn remap(key: &Self) -> [u64; 2];
}

impl<T: AsRef<[u8]>> Remap for T {
    fn remap(key: &Self) -> [u64; 2] {
        let spooky = spooky_short(key.as_ref(), 0);
        [spooky[0], spooky[1]]
    }
}

pub struct Function<T: Remap> {
    seed: u64,
    num_keys: usize,
    segment_size: usize,
    values: Box<[u64]>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Remap> Function<T> {
    pub fn get_by_sig(&self, sig: &[u64; 2]) -> u64 {
        let edge = edge(&sig, self.segment_size);
        self.values[edge[0]] ^ self.values[edge[1]] ^ self.values[edge[2]]
    }

    #[inline(always)]
    pub fn get(&self, key: &T) -> u64 {
        self.get_by_sig(&T::remap(key))
    }

    pub fn new<I: std::iter::Iterator<Item = T>, V: std::iter::Iterator<Item = u64>>(
        keys: I,
        values: &mut V,
        bit_width: usize,
        pl: &mut Option<&mut ProgressLogger>,
    ) -> Function<T> {
        pl.as_mut().map(|pl| pl.start("Reading input..."));
        let sigs = keys
            .map(|x| (T::remap(&x), values.next().unwrap()))
            .collect::<Vec<_>>();
        pl.as_mut().map(|pl| pl.done());

        let mut num_vertices = (sigs.len() as f64 * 1.12).ceil() as usize + 1;
        num_vertices = num_vertices + 130 - num_vertices % 130;
        let segment_size = num_vertices / 130;

        pl.as_mut().map(|pl| pl.start("Generating graph..."));
        let mut edge_lists = Vec::new();
        edge_lists.resize_with(num_vertices, || EdgeList::default());
        sigs.par_iter().enumerate().for_each(|(edge_index, sig)| {
            for &v in edge(&sig.0, segment_size).iter() {
                edge_lists[v].add(edge_index);
            }
        });
        pl.as_mut().map(|pl| pl.done_with_count(sigs.len()));

        let mut values = Vec::new();
        values.resize_with(num_vertices, || AtomicU64::new(0));
        let mut peeled = Vec::new();
        peeled.resize_with(sigs.len(), || AtomicBool::new(false));

        let num_peeled = AtomicUsize::new(0);
        let next = AtomicUsize::new(0);
        let incr = 1024;

        pl.as_mut().map(|pl| pl.start("Peeling and assigning..."));

        thread::scope(|s| {
            for _ in 0..1 {
                s.spawn(|| {
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

                                let (d, edge_index) = edge_lists[v].dec();
                                if d != 1 {
                                    debug_assert_eq!(d, 0, "v = {}", v);
                                    continue; // Skip no longer useful entries
                                }
                                if peeled[edge_index].swap(true, Ordering::Relaxed) {
                                    // Someone already peeled this edge
                                    continue;
                                }

                                stack[curr] = v;
                                curr += 1;
                                // Degree is necessarily 0
                                for &x in edge(&sigs[edge_index].0, segment_size).iter() {
                                    if x != v {
                                        if let (2, _) = edge_lists[x].remove(edge_index) {
                                            // Vertex transitioned to degree 1, so we put it on the stack
                                            stack.push(x);
                                        }
                                    }
                                }
                            }
                            stack.truncate(curr);
                        }
                    }
                    num_peeled.fetch_add(stack.len(), Relaxed);
                    while let Some(v) = stack.pop() {
                        let edge_index = edge_lists[v].edge_index();
                        let edge = edge(&sigs[edge_index].0, segment_size);

                        let value = if v == edge[0] {
                            values[edge[1]].load(Relaxed) ^ values[edge[2]].load(Relaxed)
                        } else if v == edge[1] {
                            values[edge[0]].load(Relaxed) ^ values[edge[2]].load(Relaxed)
                        } else {
                            debug_assert_eq!(v, edge[2]);
                            values[edge[0]].load(Relaxed) ^ values[edge[1]].load(Relaxed)
                        };
                        values[v].store(&sigs[edge_index].1 ^ value, Relaxed);
                    }
                });
            }
        });

        pl.as_mut().map(|pl| pl.done_with_count(sigs.len()));

        assert_eq!(sigs.len(), num_peeled.load(Relaxed));

        Function {
            seed: 0,
            num_keys: sigs.len(),
            segment_size,
            values: unsafe {
                std::mem::transmute::<Vec<AtomicU64>, Vec<u64>>(values).into_boxed_slice()
            },
            _phantom: std::marker::PhantomData,
        }
    }
}

#[derive(Debug, Default)]

struct EdgeList(AtomicUsize);
impl EdgeList {
    const DEG_SHIFT: usize = 8 * size_of::<usize>() - 10;
    const EDGE_INDEX_MASK: usize = (1_usize << EdgeList::DEG_SHIFT) - 1;
    const DEG: usize = 1_usize << EdgeList::DEG_SHIFT;

    #[inline(always)]
    fn add(&self, edge: usize) -> (usize, usize) {
        let t = self.0.fetch_add(EdgeList::DEG | edge, Relaxed);
        (t >> EdgeList::DEG_SHIFT, t & EdgeList::EDGE_INDEX_MASK)
    }

    #[inline(always)]
    fn remove(&self, edge: usize) -> (usize, usize) {
        let t = self.0.fetch_sub(EdgeList::DEG | edge, Relaxed);
        (t >> EdgeList::DEG_SHIFT, t & EdgeList::EDGE_INDEX_MASK)
    }

    #[inline(always)]
    fn degree(&self) -> usize {
        self.0.load(Relaxed) >> EdgeList::DEG_SHIFT
    }

    #[inline(always)]
    fn edge_index(&self) -> usize {
        self.0.load(Relaxed) & EdgeList::EDGE_INDEX_MASK
    }

    #[inline(always)]
    fn dec(&self) -> (usize, usize) {
        let t = self.0.fetch_sub(EdgeList::DEG, Relaxed);
        (t >> EdgeList::DEG_SHIFT, t & EdgeList::EDGE_INDEX_MASK)
    }
}

fn count_sort<T: Copy + Clone, F: Fn(&T) -> usize>(
    sigs: &mut [T],
    num_keys: usize,
    key: F,
) -> (Vec<usize>, Vec<usize>) {
    let mut counts = Vec::new();
    counts.resize(num_keys, 0_usize);

    for sig in &*sigs {
        counts[key(sig)] += 1;
    }

    let mut cum = Vec::new();
    cum.resize(counts.len() + 1, 0_usize);
    for i in 1..cum.len() {
        cum[i] += cum[i - 1] + counts[i - 1];
    }
    let end = sigs.len() - counts.last().unwrap();

    let mut pos = cum[1..].to_vec();
    let mut i = 0;
    while i < end {
        let mut sig = sigs[i];

        loop {
            let slot = key(&sig);
            pos[slot] -= 1;
            if pos[slot] <= i {
                break;
            }
            mem::swap(&mut sigs[pos[slot]], &mut sig);
        }
        sigs[i] = sig;
        i += counts[key(&sig)];
    }

    (counts, cum)
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
