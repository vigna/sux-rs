/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Find the minimum `c` for which a random fuse 3-hypergraph peels
//! successfully, for each `n` in `[1 . . 800000]`.
//!
//! For each `n`, starting from the previous `c` (or 1.4 for `n = 1`),
//! we lower `c` by 0.05 until the success rate drops below a
//! threshold that depends on `n`:
//!
//! | Range              | Required success rate |
//! |--------------------|-----------------------|
//! | `n ≤ 1000`         | 50%                   |
//! | `1000 < n ≤ 10000` | 75%                   |
//! | `10000 < n ≤ 100000` | 87.5%               |
//! | `100000 < n ≤ 800000` | 100%               |
//!
//! The `log2_seg_size` formula is the standard fuse formula from
//! "Binary Fuse Filters":
//! `floor(ln(n) / ln(3.33) + 2.25)`.
//!
//! Output: one line per `n` with `n`, the minimum `c` found, and
//! the success rate at that `c`.
//!
//! Usage:
//!   cargo run --release --example find_min_c [start] [end] [trials]
//!
//! Defaults: start=1, end=800000, trials=100

use rand::SeedableRng;
use rand::rngs::SmallRng;
use rayon::prelude::*;

fn log2_seg_size(n: usize) -> u32 {
    let n = n.max(1) as f64;
    (n.ln() / (3.33_f64).ln() + 2.25).floor().max(1.0) as u32
}

fn fuse_dims(n: usize, c: f64) -> (usize, usize, u32) {
    let log2_seg = log2_seg_size(n);
    let seg_size = 1usize << log2_seg;
    let target_vertices = (c * n as f64).ceil() as usize;
    let l = target_vertices.div_ceil(seg_size).saturating_sub(2).max(1);
    let num_vertices = (l + 2) * seg_size;
    (num_vertices, l, log2_seg)
}

fn gen_fuse_graph(n: usize, l: usize, log2_seg: u32, seed: u64) -> Vec<[u32; 3]> {
    use rand::RngExt;
    let seg_size = 1u64 << log2_seg;
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut edges = Vec::with_capacity(n);
    for _ in 0..n {
        let s: u64 = rng.random_range(0..l as u64);
        let o0: u64 = rng.random_range(0..seg_size);
        let o1: u64 = rng.random_range(0..seg_size);
        let o2: u64 = rng.random_range(0..seg_size);
        edges.push([
            (s * seg_size + o0) as u32,
            ((s + 1) * seg_size + o1) as u32,
            ((s + 2) * seg_size + o2) as u32,
        ]);
    }
    edges
}

fn peel(num_vertices: usize, edges: &[[u32; 3]]) -> bool {
    let n = edges.len();
    let mut edge_xor = vec![0u32; num_vertices];
    let mut degree = vec![0u8; num_vertices];

    for (i, &[a, b, c]) in edges.iter().enumerate() {
        let stored = (i + 1) as u32;
        for &v in &[a, b, c] {
            let (d, overflow) = degree[v as usize].overflowing_add(1);
            if overflow {
                return false;
            }
            degree[v as usize] = d;
            edge_xor[v as usize] ^= stored;
        }
    }

    let mut stack: Vec<u32> = (0..num_vertices as u32)
        .filter(|&v| degree[v as usize] == 1)
        .collect();

    let mut n_peeled = 0usize;
    while let Some(v) = stack.pop() {
        let vu = v as usize;
        if degree[vu] != 1 {
            continue;
        }
        let stored = edge_xor[vu];
        if stored == 0 {
            break;
        }
        n_peeled += 1;
        let [a, b, c] = edges[(stored - 1) as usize];
        for &u in &[a, b, c] {
            let uu = u as usize;
            degree[uu] -= 1;
            edge_xor[uu] ^= stored;
            if degree[uu] == 1 {
                stack.push(u);
            }
        }
    }

    n_peeled == n
}

fn required_success_rate(n: usize) -> f64 {
    if n <= 1000 {
        0.50
    } else if n <= 10000 {
        0.75
    } else if n <= 100000 {
        0.875
    } else {
        1.0
    }
}

fn try_c(n: usize, c: f64, trials: usize) -> f64 {
    let (num_vertices, l, log2_seg) = fuse_dims(n, c);
    let successes: usize = (0..trials)
        .into_par_iter()
        .filter(|&t| {
            let seed = (n as u64)
                .wrapping_mul(0x9e3779b97f4a7c15)
                .wrapping_add(t as u64);
            let edges = gen_fuse_graph(n, l, log2_seg, seed);
            peel(num_vertices, &edges)
        })
        .count();
    successes as f64 / trials as f64
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let start: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1);
    let end: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(800_000);
    let trials: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(100);

    eprintln!("find_min_c: n={start}..={end}, {trials} trials per (n, c)");
    println!("n\tc\tsuccess_rate");

    let step: f64 = 0.005;

    for n in start..=end {
        if n == 0 {
            println!("0\t0.000\t1.000");
            continue;
        }

        let threshold = required_success_rate(n);

        // Start from 2.0 and descend until we fail.
        let mut best_c: f64 = 2.0;
        let mut c = 2.0 - step;
        while c >= 0.88 {
            if try_c(n, c, trials) >= threshold {
                best_c = c;
                c -= step;
            } else {
                break;
            }
        }

        let rate = try_c(n, best_c, trials);
        println!("{n}\t{best_c:.3}\t{rate:.3}");

        if n % 10000 == 0 {
            eprintln!("n={n}\tc={best_c:.3}");
        }
    }
}
