use anyhow::Result;
use clap::Parser;
use dsi_progress_logger::ProgressLogger;
use std::io::{BufRead, BufReader};
use std::mem::{self, size_of};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::thread;
use sux::spooky::spooky_short;
use sux::spooky::spooky_short_mix;
use sux::spooky::SC_CONST;
use Ordering::Relaxed;

#[derive(Parser, Debug)]
#[command(about = "Benchmarks compact arrays", long_about = None)]
struct Args {
    filename: String,
}

const WORDS_PER_SUPERBLOCK: usize = 32;

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

pub struct MinimalPerfectHashFunction<T: Remap> {
    seed: u64,
    num_keys: usize,
    delim_seed: Vec<DelimSeed>,
    values: Values,
    counts: Vec<u64>,
    high_bits: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: AsRef<[u8]>> Remap for T {
    fn remap(key: &Self) -> [u64; 2] {
        let spooky = spooky_short(key.as_ref(), 0);
        [spooky[0], spooky[1]]
    }
}

impl<T: Remap> MinimalPerfectHashFunction<T> {
    pub fn get_by_sig(&self, sig: &[u64; 2]) -> usize {
        let chunk = (sig[0] >> (64 - self.high_bits)) as usize;
        let segment_size = self.delim_seed[chunk + 1].delim() - self.delim_seed[chunk].delim();
        let start_vertex = self.delim_seed[chunk].delim() * 130;

        let mut edge = edge(&sig, segment_size);
        edge.iter_mut().for_each(|x| *x += start_vertex);

        let mut hinge =
            edge[((self.values.get(edge[0]) + self.values.get(edge[1]) + self.values.get(edge[2]))
                % 3) as usize];

        // rank
        hinge *= 2;

        // TODO: /-% 6
        let mut word = hinge / 64;
        let block = word / (WORDS_PER_SUPERBLOCK / 2) & !1;
        let offset = ((word % WORDS_PER_SUPERBLOCK) / 6) as isize - 1;

        let mut result = self.counts[block]
            + (self.counts[block + 1]
                >> 12 * (offset + (((offset as usize) >> 32 - 4) as isize & 6))
                & 0x7FF)
            + count_nonzero_pairs(self.values.0[word] & (1_u64 << (hinge % 64)) - 1);

        for _ in 0..(word & 0x1F) % 6 {
            word -= 1;
            result += count_nonzero_pairs(self.values.0[word]);
        }

        result as usize
    }

    #[inline(always)]
    pub fn get(&self, key: &T) -> usize {
        self.get_by_sig(&T::remap(key))
    }

    pub fn new<I: std::iter::Iterator<Item = T>>(
        keys: I,
        pl: &mut Option<&mut ProgressLogger>,
    ) -> MinimalPerfectHashFunction<T> {
        pl.as_mut().map(|pl| pl.start("Reading input..."));
        let mut sigs = keys.map(|x| T::remap(&x)).collect::<Vec<_>>();

        let high_bits = 2;
        pl.as_mut().map(|pl| pl.done());

        pl.as_mut().map(|pl| pl.start("Sorting..."));
        let (counts, cum) = count_sort(&mut sigs, 1 << high_bits, |sig| {
            (sig[0] >> (64 - high_bits)) as usize
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
        let values = AtomicValues::new(num_vertices);

        let num_peeled = AtomicUsize::new(0);
        let next_chunk = AtomicUsize::new(0);

        thread::scope(|s| {
            for _ in 0..8 {
                s.spawn(|| loop {
                    let chunk = next_chunk.fetch_add(1, Relaxed);
                    if chunk >= 1 << high_bits {
                        break;
                    }
                    let (start_edge, end_edge) = (cum[chunk], cum[chunk + 1]);
                    let sigs = &sigs[start_edge..end_edge];
                    let segment_size = delims[chunk + 1] - delims[chunk];
                    let num_vertices = segment_size * 130;

                    let mut edge_lists = Vec::new();
                    edge_lists.resize(num_vertices, EdgeList::default());

                    for (edge_index, sig) in sigs.iter().enumerate() {
                        for v in edge(sig, segment_size).iter() {
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

                            for &x in edge(&sigs[edge_index], segment_size).iter() {
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

                    num_peeled.fetch_add(stack.len(), Relaxed);
                    let start_vertex = delims[chunk] * 130;

                    while let Some(v) = stack.pop() {
                        let edge_index = edge_lists[v].0; // Degree already masked out
                        let edge = edge(&sigs[edge_index], segment_size);

                        let (s, hinge_index) = if v == edge[0] {
                            (
                                values.get(edge[1] + start_vertex)
                                    + values.get(edge[2] + start_vertex),
                                0,
                            )
                        } else if v == edge[1] {
                            (
                                values.get(edge[0] + start_vertex)
                                    + values.get(edge[2] + start_vertex),
                                1,
                            )
                        } else {
                            debug_assert_eq!(v, edge[2]);
                            (
                                values.get(edge[0] + start_vertex)
                                    + values.get(edge[1] + start_vertex),
                                2,
                            )
                        };
                        let value = (9 + hinge_index - s) % 3;
                        values.set(v + start_vertex, if value == 0 { 3 } else { value });
                    }
                });
            }
        });

        pl.as_mut().map(|pl| pl.done_with_count(sigs.len()));

        assert_eq!(sigs.len(), num_peeled.load(Relaxed));

        assert_eq!(
            sigs.len(),
            sigs.iter()
                .map(|sig| {
                    let chunk = (sig[0] >> (64 - high_bits)) as usize;
                    let segment_size = delims[chunk + 1] - delims[chunk];
                    let start_vertex = delims[chunk] * 130;
                    let mut edge = edge(sig, segment_size);
                    edge.iter_mut().for_each(|x| *x += start_vertex);

                    edge[((values.get(edge[0]) + values.get(edge[1]) + values.get(edge[2])) % 3)
                        as usize]
                })
                .collect::<std::collections::HashSet<_>>()
                .len()
        );

        pl.as_mut().map(|pl| pl.start("Build rank..."));
        let words = &values.0;
        let num_counts =
            ((num_vertices * 2 + WORDS_PER_SUPERBLOCK * 64 - 1) / (WORDS_PER_SUPERBLOCK * 64)) * 2;
        // Init rank/select structure
        let mut counts = Vec::new();
        counts.resize(num_counts + 1, 0);

        let mut c = 0;
        let mut pos = 0;
        let mut i = 0;
        while i < words.len() {
            counts[pos] = c;
            for j in 0..WORDS_PER_SUPERBLOCK {
                if j != 0 && j % 6 == 0 {
                    counts[pos + 1] |= if i + j <= words.len() {
                        c - counts[pos]
                    } else {
                        0x7FF
                    } << 12 * (j / 6 - 1);
                }
                if i + j < words.len() {
                    c += count_nonzero_pairs(words[i + j].load(Relaxed));
                }
            }
            i += WORDS_PER_SUPERBLOCK;
            pos += 2;
        }

        pl.as_mut().map(|pl| pl.done_with_count(sigs.len()));

        MinimalPerfectHashFunction {
            seed: 0,
            num_keys: sigs.len(),
            delim_seed: delims
                .into_iter()
                .map(|x| DelimSeed(x as u64))
                .collect::<Vec<_>>(),
            values: values.into(),
            counts,
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
    let mph = MinimalPerfectHashFunction::new(
        BufReader::new(file).lines().map(|line| line.unwrap()),
        &mut Some(&mut pl),
    );

    let file = std::fs::File::open(&args.filename)?;
    let keys = BufReader::new(file)
        .lines()
        .map(|line| line.unwrap())
        .take(10_000_000)
        .collect::<Vec<_>>();

    pl.start("Querying...");
    for key in &keys {
        std::hint::black_box(mph.get(key));
    }
    pl.done_with_count(keys.len());

    let file = std::fs::File::open(&args.filename)?;
    let keys = BufReader::new(file).lines().map(|line| line.unwrap());

    let mut out = <std::collections::HashSet<_>>::new();
    let mut c: usize = 0;
    keys.for_each(|key| {
        let result = mph.get(&key);
        out.insert(result);
        c += 1;
    });
    assert_eq!(c, out.len());

    Ok(())
}
