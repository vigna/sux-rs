use anyhow::Result;
use clap::Parser;
use std::io::{BufRead, BufReader};
use std::mem::size_of;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::thread;
use std::time::Instant;
use sux::prelude::CompactArray;
use sux::prelude::*;
use sux::spooky::spooky_short;
use sux::spooky::spooky_short_mix;
use sux::spooky::SC_CONST;
use Ordering::Relaxed;

#[derive(Parser, Debug)]
#[command(about = "Benchmarks compact arrays", long_about = None)]
struct Args {
    filename: String,
}
const DEG_SHIFT: usize = 8 * size_of::<usize>() - 10;
const DEG: usize = 1_usize << DEG_SHIFT;

fn count_nonzero_pairs(x: u64) -> usize {
    ((x | x >> 1) & 0x5555555555555555).count_ones() as usize
}

struct Values(Vec<AtomicU64>);
impl Values {
    fn new(n: usize) -> Self {
        let mut v = Vec::new();
        v.resize_with((n * 2 + 63) / 64, || AtomicU64::new(0));
        Self(v)
    }
    fn get(&self, i: usize) -> u64 {
        let pos = i * 2;
        self.0[pos / 64].load(Relaxed) >> (pos % 64) & 3
    }

    fn set(&self, i: usize, v: u64) {
        let pos = i * 2;
        self.0[pos / 64].fetch_or(v << pos % 64, Relaxed);
    }
}

fn main() -> Result<()> {
    stderrlog::new()
        .verbosity(2)
        .timestamp(stderrlog::Timestamp::Second)
        .init()
        .unwrap();

    let args = Args::parse();

    let start = Instant::now();

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

    let file = std::fs::File::open(args.filename)?;
    let sigs = BufReader::new(file)
        .lines()
        .map(|line| {
            let sig = spooky_short(line.unwrap().as_bytes(), 0);
            [sig[0], sig[1]]
        })
        .collect::<Vec<_>>();

    println!("{}", start.elapsed().as_nanos() / sigs.len() as u128);

    let start = Instant::now();

    let num_edges = sigs.len();
    let mut num_vertices = (num_edges as f64 * 1.12).ceil() as usize + 1;
    num_vertices = num_vertices + 130 - num_vertices % 130;
    let segment_size = num_vertices / 130;

    let mut deg_add = Vec::new();
    deg_add.resize_with(num_vertices, || AtomicUsize::new(0));
    let mut peeled = Vec::new();
    peeled.resize_with(num_edges, || AtomicBool::new(false));
    let mut values = Values::new(num_vertices);

    for (edge_index, sig) in sigs.iter().enumerate() {
        let tuple = spooky_short_rehash(sig, 0);
        let first_segment = tuple[3] as usize % 128;
        for i in 0..3 {
            let vertex = ((tuple[i] as u128) * (segment_size as u128) >> 64) as usize
                + (first_segment + i) * segment_size;
            // Djamal's trick: xor edges incident to a vertex.
            // We will be visiting only vertices of degree 1 anyway.
            deg_add[vertex].fetch_add(DEG | edge_index, Relaxed);
        }
    }

    let mut num_peeled = AtomicUsize::new(0);
    let curr = AtomicUsize::new(0);
    thread::scope(|s| {
        for _ in 0..4 {
            s.spawn(|| {
                let mut stack = Vec::new();
                loop {
                    let next = curr.fetch_add(1, Relaxed);
                    if next >= num_vertices {
                        break;
                    }
                    if deg_add[next].load(Relaxed) >> DEG_SHIFT != 1 {
                        continue;
                    }

                    let mut pos = stack.len();
                    let mut curr = stack.len();
                    // Stack initialization
                    stack.push(next);

                    while pos < stack.len() {
                        let v = stack[pos];
                        pos += 1;

                        let t = deg_add[v].load(Relaxed);
                        let d = t >> DEG_SHIFT;
                        if d != 1 {
                            assert_eq!(d, 0, "v = {}", v);

                            continue; // Skip no longer useful entries
                        }
                        let edge_index = t & (1_usize << DEG_SHIFT) - 1;
                        if peeled[edge_index].swap(true, Relaxed) {
                            continue;
                        }
                        deg_add[v].store(edge_index, Relaxed);

                        stack[curr] = v;
                        curr += 1;
                        let tuple = spooky_short_rehash(&sigs[edge_index], 0);
                        let first_segment = tuple[3] as usize % 128;
                        let edge = [
                            ((tuple[0] as u128) * (segment_size as u128) >> 64) as usize
                                + (first_segment + 0) * segment_size,
                            ((tuple[1] as u128) * (segment_size as u128) >> 64) as usize
                                + (first_segment + 1) * segment_size,
                            ((tuple[2] as u128) * (segment_size as u128) >> 64) as usize
                                + (first_segment + 2) * segment_size,
                        ];

                        for i in 0..3 {
                            if edge[i] != v {
                                if deg_add[edge[i]].fetch_sub(DEG | edge_index, Relaxed)
                                    >> DEG_SHIFT
                                    == 2
                                {
                                    stack.push(edge[i]);
                                }
                            }
                        }
                    }

                    stack.truncate(curr);

                    num_peeled.fetch_add(stack.len(), Relaxed);

                    while let Some(v) = stack.pop() {
                        let edge_index = deg_add[v].load(Relaxed) >> DEG_SHIFT;
                        let tuple = spooky_short_rehash(&sigs[edge_index], 0);
                        let first_segment = tuple[3] as usize % 128;
                        let edge = [
                            ((tuple[0] as u128) * (segment_size as u128) >> 64) as usize
                                + (first_segment + 0) * segment_size,
                            ((tuple[1] as u128) * (segment_size as u128) >> 64) as usize
                                + (first_segment + 1) * segment_size,
                            ((tuple[2] as u128) * (segment_size as u128) >> 64) as usize
                                + (first_segment + 2) * segment_size,
                        ];

                        let (s, hinge_index) = if v == edge[0] {
                            (values.get(edge[1]) + values.get(edge[2]), 0)
                        } else if v == edge[1] {
                            (values.get(edge[0]) + values.get(edge[2]), 1)
                        } else {
                            debug_assert_eq!(v, edge[2]);
                            (values.get(edge[0]) + values.get(edge[1]), 2)
                        };
                        let value = (9 + hinge_index - s) % 3;
                        values.set(v, if value == 0 { 3 } else { value });
                    }
                }
            });
        }
    });

    for i in 0..num_vertices {
        assert_ne!(deg_add[i].load(Relaxed) >> DEG_SHIFT, 1, "v = {}", i);
    }
    assert_eq!(num_edges, num_peeled.load(Relaxed));
    println!("{}", start.elapsed().as_nanos() / sigs.len() as u128);

    println!("{}", start.elapsed().as_nanos() / sigs.len() as u128);

    let words = &values.0;
    const WORDS_PER_SUPERBLOCK: usize = 32;
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

    let mut out = Vec::new();

    let start = Instant::now();

    for sig in &sigs {
        let tuple = spooky_short_rehash(sig, 0);
        let first_segment = tuple[3] as usize % 128;
        let edge = [
            ((tuple[0] as u128) * (segment_size as u128) >> 64) as usize
                + (first_segment + 0) * segment_size,
            ((tuple[1] as u128) * (segment_size as u128) >> 64) as usize
                + (first_segment + 1) * segment_size,
            ((tuple[2] as u128) * (segment_size as u128) >> 64) as usize
                + (first_segment + 2) * segment_size,
        ];

        let mut hinge =
            edge[((values.get(edge[0]) + values.get(edge[1]) + values.get(edge[2])) % 3) as usize];

        // rank
        hinge *= 2;

        let mut word = hinge / 64;
        let block = word / (WORDS_PER_SUPERBLOCK / 2) & !1;
        let offset = ((word % WORDS_PER_SUPERBLOCK) / 6) as isize - 1;

        let mut result = counts[block]
            + (counts[block + 1] >> 12 * (offset + (((offset as usize) >> 32 - 4) as isize & 6))
                & 0x7FF)
            + count_nonzero_pairs(words[word].load(Relaxed) & (1_u64 << (hinge % 64)) - 1);

        for _ in 0..(word & 0x1F) % 6 {
            word -= 1;
            result += count_nonzero_pairs(words[word].load(Relaxed));
        }

        assert!(result < num_edges);
        out.push(result);
    }

    println!("{}", start.elapsed().as_nanos() / sigs.len() as u128);

    assert_eq!(
        out.len(),
        out.into_iter()
            .collect::<std::collections::HashSet::<_>>()
            .len()
    );

    Ok(())
}
