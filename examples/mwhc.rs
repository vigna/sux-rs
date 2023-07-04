use anyhow::Result;
use clap::Parser;
use std::io::{BufRead, BufReader};
use std::sync::atomic::{AtomicU16, AtomicUsize, Ordering};
use std::time::Instant;
use sux::prelude::CompactArray;
use sux::prelude::*;
use sux::spooky::spooky_short;
use sux::spooky::spooky_short_mix;
use sux::spooky::SC_CONST;

#[derive(Parser, Debug)]
#[command(about = "Benchmarks compact arrays", long_about = None)]
struct Args {
    filename: String,
}

fn count_nonzero_pairs(x: u64) -> usize {
    ((x | x >> 1) & 0x5555555555555555).count_ones() as usize
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

    let mut deg = Vec::new();
    deg.resize_with(num_vertices, || AtomicU16::new(0));
    let mut xor = Vec::new();
    xor.resize_with(num_vertices, || AtomicUsize::new(0));
    let mut stack = Vec::new();

    for (edge_index, sig) in sigs.iter().enumerate() {
        let tuple = spooky_short_rehash(sig, 0);
        let first_segment = tuple[3] as usize % 128;
        for i in 0..3 {
            let vertex = ((tuple[i] as u128) * (segment_size as u128) >> 64) as usize
                + (first_segment + i) * segment_size;
            // Djamal's trick: xor edges incident to a vertex.
            // We will be visiting only vertices of degree 1 anyway.
            xor[vertex].fetch_xor(edge_index, Ordering::Relaxed);
            deg[vertex].fetch_add(1, Ordering::Relaxed);
        }
    }

    for x in 0..num_vertices {
        if deg[x].load(Ordering::Relaxed) != 1 {
            continue;
        }

        let mut pos = stack.len();
        let mut curr = stack.len();
        // Stack initialization
        stack.push(x);

        while pos < stack.len() {
            let v = stack[pos];
            pos += 1;

            if deg[v].load(Ordering::Relaxed) != 1 {
                continue; // Skip no longer useful entries
            }

            let edge_index = xor[v].swap(usize::MAX, Ordering::Relaxed);
            if edge_index == usize::MAX {
                debug_assert_eq!(deg[1].load(Ordering::Relaxed), 0);
                continue; // Skip no longer useful entries
            }

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
                if deg[edge[i]].fetch_sub(1, Ordering::Relaxed) == 2 {
                    stack.push(edge[i]);
                }
                // When edge[i] == v this is useless, but we avoid a branch.
                xor[edge[i]].fetch_xor(edge_index, Ordering::Relaxed);
            }

            xor[v].store(edge_index, Ordering::Relaxed);
        }

        stack.truncate(curr);
    }

    assert_eq!(num_edges, stack.len());

    let mut values = CompactArray::new(2, num_vertices);

    while let Some(v) = stack.pop() {
        let edge_index = xor[v].load(Ordering::Relaxed);
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

    println!("{}", start.elapsed().as_nanos() / sigs.len() as u128);

    let words = values.as_ref();
    const WORDS_PER_SUPERBLOCK: usize = 32;
    let num_counts =
        ((values.len() * 2 + WORDS_PER_SUPERBLOCK * 64 - 1) / (WORDS_PER_SUPERBLOCK * 64)) * 2;
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
                c += count_nonzero_pairs(words[i + j]);
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
            + count_nonzero_pairs(words[word] & (1_u64 << (hinge % 64)) - 1);

        for _ in 0..(word & 0x1F) % 6 {
            word -= 1;
            result += count_nonzero_pairs(words[word]);
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
