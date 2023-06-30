use anyhow::Result;
use clap::Parser;
use std::io::{BufRead, BufReader};
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
    let num_vertices = (num_edges as f64 * 1.23).ceil() as usize + 1;

    let mut deg = Vec::new();
    deg.resize(num_vertices, 0);
    let mut xor = Vec::new();
    xor.resize(num_vertices, 0);
    let mut stack = Vec::new();

    for (edge_index, sig) in sigs.iter().enumerate() {
        let tuple = spooky_short_rehash(sig, 0);
        for i in 0..3 {
            let vertex = ((tuple[i] as u128) * (num_vertices as u128) >> 64) as usize;
            // Djamal's trick: xor edges incident to a vertex.
            // We will be visiting only vertices of degree 1 anyway.
            xor[vertex] ^= edge_index;
            deg[vertex] += 1;
        }
    }

    for x in 0..num_vertices {
        if deg[x] != 1 {
            continue;
        }

        let mut pos = stack.len();
        let mut curr = stack.len();
        // Stack initialization
        stack.push(x);

        while pos < stack.len() {
            let v = stack[pos];
            pos += 1;

            if deg[v] != 1 {
                continue; // Skip no longer useful entries
            }

            stack[curr] = v;
            curr += 1;

            let edge_index = xor[v];

            let tuple = spooky_short_rehash(&sigs[edge_index], 0);
            let edge = [
                ((tuple[0] as u128) * (num_vertices as u128) >> 64) as usize,
                ((tuple[1] as u128) * (num_vertices as u128) >> 64) as usize,
                ((tuple[2] as u128) * (num_vertices as u128) >> 64) as usize,
            ];

            for i in 0..3 {
                deg[edge[i]] -= 1;
                if edge[i] != v {
                    xor[edge[i]] ^= edge_index;
                }
            }
            if deg[edge[0]] == 1 {
                stack.push(edge[0]);
            }
            if deg[edge[1]] == 1 && edge[1] != edge[0] {
                stack.push(edge[1]);
            }
            if deg[edge[2]] == 1 && edge[2] != edge[0] && edge[2] != edge[1] {
                stack.push(edge[2]);
            }
        }

        stack.truncate(curr);
    }

    assert_eq!(num_edges, stack.len());

    let mut values = CompactArray::new(2, num_vertices);

    while let Some(v) = stack.pop() {
        let edge_index = xor[v];
        let tuple = spooky_short_rehash(&sigs[edge_index], 0);
        let edge = [
            ((tuple[0] as u128) * (num_vertices as u128) >> 64) as usize,
            ((tuple[1] as u128) * (num_vertices as u128) >> 64) as usize,
            ((tuple[2] as u128) * (num_vertices as u128) >> 64) as usize,
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
        let edge = [
            ((tuple[0] as u128) * (num_vertices as u128) >> 64) as usize,
            ((tuple[1] as u128) * (num_vertices as u128) >> 64) as usize,
            ((tuple[2] as u128) * (num_vertices as u128) >> 64) as usize,
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
