use anyhow::Result;
use clap::Parser;
use dsi_progress_logger::ProgressLogger;
use std::io::{BufRead, BufReader};
use std::mem::{self, size_of};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
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
const EDGE_INDEX_MASK: usize = (1_usize << DEG_SHIFT) - 1;
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

    let mut pl = ProgressLogger::default();

    pl.start("Reading input...");
    let file = std::fs::File::open(args.filename)?;
    let mut sigs = BufReader::new(file)
        .lines()
        .map(|line| {
            let sig = spooky_short(line.unwrap().as_bytes(), 0);
            [sig[0], sig[1]]
        })
        .collect::<Vec<_>>();

    let high_bits = 2;
    pl.done();

    pl.start("Sorting...");

    let mut counts = Vec::new();
    counts.resize(1 << high_bits, 0_usize);

    for sig in &sigs {
        counts[(sig[0] >> (64 - high_bits)) as usize] += 1;
    }

    let mut cum = Vec::new();
    cum.resize(counts.len() + 1, 0_usize);
    for i in 1..cum.len() {
        cum[i] += cum[i - 1] + counts[i - 1];
    }
    let end = sigs.len() - counts.last().unwrap();

    let mut delims = vec![0];
    for &count in &counts {
        let mut num_vertices = (count as f64 * 1.12).ceil() as usize + 1;
        num_vertices = num_vertices + 130 - num_vertices % 130;
        delims.push(delims.last().unwrap() + num_vertices / 130);
    }

    dbg!(&counts, &cum, &delims);

    let mut pos = cum[1..].to_vec();
    let mut i = 0;
    while i < end {
        let mut sig = sigs[i];

        loop {
            let slot = (sig[0] >> (64 - high_bits)) as usize;
            pos[slot] -= 1;
            if pos[slot] <= i {
                break;
            }
            mem::swap(&mut sigs[pos[slot]], &mut sig);
        }
        sigs[i] = sig;
        i += counts[(sig[0] >> (64 - high_bits)) as usize];
    }

    pl.done_with_count(sigs.len());

    for i in 1..sigs.len() {
        assert!(
            (sigs[i - 1][0] >> (64 - high_bits)) as usize
                <= (sigs[i][0] >> (64 - high_bits)) as usize
        );
    }

    pl.start("Peeling graph...");
    let num_vertices = (*cum.last().unwrap() as f64 * 1.12) as usize;
    let mut values = Values::new(*delims.last().unwrap() * 130);

    let mut num_peeled = AtomicUsize::new(0);
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
                let num_edges = sigs.len();
                let segment_size = delims[chunk + 1] - delims[chunk];
                let mut num_vertices = segment_size * 130;

                let mut deg_add = Vec::new();
                deg_add.resize(num_vertices, 0);

                for (edge_index, sig) in sigs.iter().enumerate() {
                    let tuple = spooky_short_rehash(sig, 0);
                    let first_segment = tuple[3] as usize % 128;
                    for i in 0..3 {
                        let vertex = ((tuple[i] as u128) * (segment_size as u128) >> 64) as usize
                            + (first_segment + i) * segment_size;
                        // Djamal's trick: xor edges incident to a vertex.
                        // We will be visiting only vertices of degree 1 anyway.
                        deg_add[vertex] += DEG | edge_index;
                    }
                }

                let mut stack = Vec::new();
                let mut pos = 0;
                let mut curr = 0;

                for x in 0..num_vertices {
                    let t = deg_add[x];
                    if t >> DEG_SHIFT != 1 {
                        // degree != 1
                        continue;
                    }
                    pos = stack.len();
                    curr = stack.len();
                    stack.push(x);

                    while pos < stack.len() {
                        let v = stack[pos];
                        pos += 1;
                        if deg_add[v] >> DEG_SHIFT != 1 {
                            continue; // Skip no longer useful entries
                        }

                        deg_add[v] &= EDGE_INDEX_MASK;
                        let edge_index = deg_add[v];

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
                                deg_add[edge[i]] -= DEG | edge_index;
                                if deg_add[edge[i]] >> DEG_SHIFT == 1 {
                                    {
                                        stack.push(edge[i]);
                                    }
                                }
                            }
                        }
                    }
                    stack.truncate(curr);
                }

                num_peeled.fetch_add(stack.len(), Relaxed);
                let start_vertex = delims[chunk] * 130;

                while let Some(v) = stack.pop() {
                    let edge_index = deg_add[v];
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
                        (
                            values.get(edge[1] + start_vertex) + values.get(edge[2] + start_vertex),
                            0,
                        )
                    } else if v == edge[1] {
                        (
                            values.get(edge[0] + start_vertex) + values.get(edge[2] + start_vertex),
                            1,
                        )
                    } else {
                        debug_assert_eq!(v, edge[2]);
                        (
                            values.get(edge[0] + start_vertex) + values.get(edge[1] + start_vertex),
                            2,
                        )
                    };
                    let value = (9 + hinge_index - s) % 3;
                    values.set(v + start_vertex, if value == 0 { 3 } else { value });
                }
            });
        }
    });

    pl.done_with_count(sigs.len());

    assert_eq!(sigs.len(), num_peeled.load(Relaxed));

    assert_eq!(
        sigs.len(),
        sigs.iter()
            .map(|sig| {
                let chunk = (sig[0] >> (64 - high_bits)) as usize;
                let tuple = spooky_short_rehash(sig, 0);
                let first_segment = tuple[3] as usize % 128;
                let segment_size = delims[chunk + 1] - delims[chunk];
                let start_vertex = delims[chunk] * 130;
                let edge = [
                    ((tuple[0] as u128) * (segment_size as u128) >> 64) as usize
                        + (first_segment + 0) * segment_size
                        + start_vertex,
                    ((tuple[1] as u128) * (segment_size as u128) >> 64) as usize
                        + (first_segment + 1) * segment_size
                        + start_vertex,
                    ((tuple[2] as u128) * (segment_size as u128) >> 64) as usize
                        + (first_segment + 2) * segment_size
                        + start_vertex,
                ];

                edge[((values.get(edge[0]) + values.get(edge[1]) + values.get(edge[2])) % 3)
                    as usize]
            })
            .collect::<std::collections::HashSet<_>>()
            .len()
    );

    pl.start("Build rank...");
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

    pl.done_with_count(sigs.len());
    let mut out = Vec::new();

    let start = Instant::now();

    pl.start("Querying...");
    for sig in &sigs {
        let chunk = (sig[0] >> (64 - high_bits)) as usize;
        let tuple = spooky_short_rehash(sig, 0);
        let first_segment = tuple[3] as usize % 128;
        let segment_size = delims[chunk + 1] - delims[chunk];
        let start_vertex = delims[chunk] * 130;
        let edge = [
            ((tuple[0] as u128) * (segment_size as u128) >> 64) as usize
                + (first_segment + 0) * segment_size
                + start_vertex,
            ((tuple[1] as u128) * (segment_size as u128) >> 64) as usize
                + (first_segment + 1) * segment_size
                + start_vertex,
            ((tuple[2] as u128) * (segment_size as u128) >> 64) as usize
                + (first_segment + 2) * segment_size
                + start_vertex,
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

        assert!(result < sigs.len());
        out.push(result);
    }
    pl.done_with_count(sigs.len());

    println!("{}", start.elapsed().as_nanos() / sigs.len() as u128);

    assert_eq!(
        out.len(),
        out.into_iter()
            .collect::<std::collections::HashSet::<_>>()
            .len()
    );

    Ok(())
}
