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
use std::cell::RefCell;

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

fn gen_fuse_graph(n: usize, l: usize, log2_seg: u32, seed: u64, edges: &mut Vec<[u32; 3]>) {
    use rand::RngExt;
    let seg_mask = (1u64 << log2_seg) - 1;
    let mut rng = SmallRng::seed_from_u64(seed);
    edges.clear();
    for _ in 0..n {
        let s: u64 = rng.random_range(0..l as u64);
        let o0: u64 = rng.random::<u64>() & seg_mask;
        let o1: u64 = rng.random::<u64>() & seg_mask;
        let o2: u64 = rng.random::<u64>() & seg_mask;
        edges.push([
            ((s << log2_seg) + o0) as u32,
            (((s + 1) << log2_seg) + o1) as u32,
            (((s + 2) << log2_seg) + o2) as u32,
        ]);
    }
}

fn peel(num_vertices: usize, edges: &[[u32; 3]], bufs: &mut PeelBufs) -> bool {
    let n = edges.len();
    bufs.edge_xor.clear();
    bufs.edge_xor.resize(num_vertices, 0u32);
    bufs.degree.clear();
    bufs.degree.resize(num_vertices, 0u8);
    bufs.stack.clear();

    for (i, &[a, b, c]) in edges.iter().enumerate() {
        let stored = (i + 1) as u32;
        for &v in &[a, b, c] {
            let (d, overflow) = bufs.degree[v as usize].overflowing_add(1);
            if overflow {
                return false;
            }
            bufs.degree[v as usize] = d;
            bufs.edge_xor[v as usize] ^= stored;
        }
    }

    for v in 0..num_vertices as u32 {
        if bufs.degree[v as usize] == 1 {
            bufs.stack.push(v);
        }
    }

    let mut n_peeled = 0usize;
    while let Some(v) = bufs.stack.pop() {
        let vu = v as usize;
        if bufs.degree[vu] != 1 {
            continue;
        }
        let stored = bufs.edge_xor[vu];
        if stored == 0 {
            break;
        }
        n_peeled += 1;
        let [a, b, c] = edges[(stored - 1) as usize];
        for &u in &[a, b, c] {
            let uu = u as usize;
            bufs.degree[uu] -= 1;
            bufs.edge_xor[uu] ^= stored;
            if bufs.degree[uu] == 1 {
                bufs.stack.push(u);
            }
        }
    }

    n_peeled == n
}

struct PeelBufs {
    edge_xor: Vec<u32>,
    degree: Vec<u8>,
    stack: Vec<u32>,
}

impl PeelBufs {
    fn new() -> Self {
        Self {
            edge_xor: Vec::new(),
            degree: Vec::new(),
            stack: Vec::new(),
        }
    }
}

fn required_success_rate(n: usize) -> f64 {
    if n <= 1000 {
        0.25
    } else if n <= 10000 {
        0.50
    } else if n <= 100000 {
        0.75
    } else {
        0.9
    }
}

thread_local! {
    static BUFS: RefCell<(Vec<[u32; 3]>, PeelBufs)> = RefCell::new((Vec::new(), PeelBufs::new()));
}

fn try_c(n: usize, c: f64, trials: usize) -> f64 {
    let (num_vertices, l, log2_seg) = fuse_dims(n, c);
    let successes: usize = (0..trials)
        .into_par_iter()
        .filter(|&t| {
            BUFS.with(|cell| {
                let (edges, bufs) = &mut *cell.borrow_mut();
                gen_fuse_graph(n, l, log2_seg, t as u64, edges);
                peel(num_vertices, edges, bufs)
            })
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

    let step: f64 = 0.01;

    for n in start..=end {
        if n == 0 {
            println!("0\t0.000\t1.000");
            continue;
        }

        let threshold = required_success_rate(n);

        // Start from 1.35 and descend until we fail.
        let mut best_c: f64 = 1.35;
        let mut c = 1.35 - step;
        while c >= 1.10 {
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
