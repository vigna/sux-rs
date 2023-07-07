use anyhow::Result;
use clap::Parser;
use dsi_progress_logger::ProgressLogger;
use std::io::{BufRead, BufReader};
use std::mem::{self, size_of};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::thread;
use sux::prelude::CompactArray;
use sux::prelude::VSlice;
use sux::prelude::VSliceMut;
use sux::spooky::spooky_short;
use sux::spooky::spooky_short_mix;
use sux::spooky::SC_CONST;
use Ordering::Relaxed;

#[derive(Parser, Debug)]
#[command(about = "Benchmarks compact arrays", long_about = None)]
struct Args {
    filename: String,
}

struct DelimSeed(u64);

impl DelimSeed {
    fn new(seed: u8, delim: usize) -> Self {
        return Self((seed as u64) << 56 | (delim as u64));
    }

    fn seed(&self) -> u64 {
        self.0 >> 56
    }
    fn delim(&self) -> usize {
        (self.0 & ((1 << 56) - 1)) as usize
    }
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

fn generate_and_sort(sigs: &[([u64; 2], u64)], segment_size: usize) -> (Vec<usize>, Vec<EdgeList>) {
    let num_vertices = segment_size * 130;

    let mut edge_lists = Vec::new();
    edge_lists.resize(num_vertices, EdgeList::default());

    for (edge_index, sig) in sigs.iter().enumerate() {
        for v in edge(&sig.0, segment_size).iter() {
            edge_lists[*v].add(edge_index);
        }
    }

    let mut stack = Vec::new();
    let mut pos;
    let mut curr;

    for x in 0..num_vertices {
        let t = edge_lists[x];
        if t.degree() != 1 {
            continue;
        }
        pos = stack.len();
        curr = stack.len();
        stack.push(x);

        while pos < stack.len() {
            let v = stack[pos];
            pos += 1;
            if edge_lists[v].degree() != 1 {
                continue; // Skip no longer useful entries
            }

            let edge_index = edge_lists[v].choose();

            stack[curr] = v;
            curr += 1;

            for &x in edge(&sigs[edge_index].0, segment_size).iter() {
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

    (stack, edge_lists)
}

pub struct Function<T: Remap> {
    seed: u64,
    num_keys: usize,
    delim_seed: Vec<DelimSeed>,
    values: CompactArray<Vec<u64>>,
    high_bits: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Remap> Function<T> {
    pub fn get_by_sig(&self, sig: &[u64; 2]) -> u64 {
        let chunk = (sig[0] >> (64 - self.high_bits)) as usize;
        let segment_size = self.delim_seed[chunk + 1].delim() - self.delim_seed[chunk].delim();
        let start_vertex = self.delim_seed[chunk].delim() * 130;

        let mut edge = edge(&sig, segment_size);
        edge.iter_mut().for_each(|x| *x += start_vertex);
        self.values.get(edge[0]) ^ self.values.get(edge[1]) ^ self.values.get(edge[2])
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
        let mut sigs = keys
            .map(|x| (T::remap(&x), values.next().unwrap()))
            .collect::<Vec<_>>();

        //assert!(values.next().is_none());

        let high_bits = 2;
        pl.as_mut().map(|pl| pl.done());

        pl.as_mut().map(|pl| pl.start("Sorting..."));
        let (counts, cum) = count_sort(&mut sigs, 1 << high_bits, |sig| {
            (sig.0[0] >> (64 - high_bits)) as usize
        });
        pl.as_mut().map(|pl| pl.done_with_count(sigs.len()));

        let mut delims = vec![0];
        for &count in &counts {
            let mut num_vertices = (count as f64 * 1.12).ceil() as usize + 1;
            num_vertices = num_vertices + 130 - num_vertices % 130;
            delims.push(delims.last().unwrap() + num_vertices / 130);
        }

        pl.as_mut().map(|pl| pl.start("Peeling graph..."));
        let num_vertices = delims.last().unwrap() * 130;
        let mut values = CompactArray::new(bit_width, num_vertices);

        let num_peeled = AtomicUsize::new(0);
        let next_chunk = AtomicUsize::new(0);

        thread::scope(|s| {
            {
                s.spawn(|| loop {
                    let chunk = next_chunk.fetch_add(1, Relaxed);
                    if chunk >= 1 << high_bits {
                        break;
                    }
                    let (start_edge, end_edge) = (cum[chunk], cum[chunk + 1]);
                    let segment_size = delims[chunk + 1] - delims[chunk];
                    let sigs = &sigs[start_edge..end_edge];
                    let (mut stack, edge_lists) = generate_and_sort(sigs, segment_size);
                    num_peeled.fetch_add(stack.len(), Relaxed);
                    let start_vertex = delims[chunk] * 130;

                    while let Some(v) = stack.pop() {
                        let edge_index = edge_lists[v].0; // Degree already masked out
                        let edge = edge(&sigs[edge_index].0, segment_size);

                        let value = if v == edge[0] {
                            values.get(edge[1] + start_vertex) ^ values.get(edge[2] + start_vertex)
                        } else if v == edge[1] {
                            values.get(edge[0] + start_vertex) ^ values.get(edge[2] + start_vertex)
                        } else {
                            debug_assert_eq!(v, edge[2]);
                            values.get(edge[0] + start_vertex) ^ values.get(edge[1] + start_vertex)
                        };
                        values.set(v + start_vertex, &sigs[edge_index].1 ^ value);
                    }
                });
            }
        });

        pl.as_mut().map(|pl| pl.done_with_count(sigs.len()));

        assert_eq!(sigs.len(), num_peeled.load(Relaxed));

        Function {
            seed: 0,
            num_keys: sigs.len(),
            delim_seed: delims
                .into_iter()
                .map(|x| DelimSeed(x as u64))
                .collect::<Vec<_>>(),
            values: values.into(),
            high_bits,
            _phantom: std::marker::PhantomData,
        }
    }
}

fn count_nonzero_pairs(x: u64) -> u64 {
    ((x | x >> 1) & 0x5555555555555555).count_ones() as u64
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]

struct EdgeList(usize);
impl EdgeList {
    const DEG_SHIFT: usize = 8 * size_of::<usize>() - 10;
    const EDGE_INDEX_MASK: usize = (1_usize << EdgeList::DEG_SHIFT) - 1;
    const DEG: usize = 1_usize << EdgeList::DEG_SHIFT;

    #[inline(always)]
    fn add(&mut self, edge: usize) {
        self.0 += EdgeList::DEG;
        self.0 ^= edge;
    }

    #[inline(always)]
    fn remove(&mut self, edge: usize) {
        self.0 -= EdgeList::DEG;
        self.0 ^= edge;
    }

    #[inline(always)]
    fn degree(&self) -> usize {
        self.0 >> EdgeList::DEG_SHIFT
    }

    #[inline(always)]
    fn choose(&mut self) -> usize {
        debug_assert!(self.degree() == 1);
        self.0 &= EdgeList::EDGE_INDEX_MASK;
        self.0
    }
}

struct AtomicValues(Vec<AtomicU64>);
impl AtomicValues {
    #[inline(always)]
    fn new(n: usize) -> Self {
        let mut v = Vec::new();
        v.resize_with((n * 2 + 63) / 64, || AtomicU64::new(0));
        Self(v)
    }

    #[inline(always)]
    fn get(&self, i: usize) -> u64 {
        let pos = i * 2;
        self.0[pos / 64].load(Relaxed) >> (pos % 64) & 3
    }

    #[inline(always)]
    fn set(&self, i: usize, v: u64) {
        let pos = i * 2;
        self.0[pos / 64].fetch_or(v << pos % 64, Relaxed);
    }
}

impl Into<Values> for AtomicValues {
    fn into(self) -> Values {
        let ptr = self.0.as_ptr() as *const u64;
        Values(Box::from(unsafe {
            std::slice::from_raw_parts(ptr, self.0.len())
        }))
    }
}

struct Values(Box<[u64]>);

impl Values {
    #[inline(always)]
    fn get(&self, i: usize) -> u64 {
        let pos = i * 2;
        self.0[pos / 64] >> (pos % 64) & 3
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
    let mph = Function::new(
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
        assert_eq!(index, mph.get(key) as usize);
    }
    pl.done_with_count(keys.len());

    Ok(())
}
