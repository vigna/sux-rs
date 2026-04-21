/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Parameter sweep measuring 3-uniform fuse-graph peelability under
//! *correlated* edge generation.
//!
//! For each `(s, dist, Δlog2_seg, Δc)` cell, builds a fuse-structured
//! hypergraph where `s` "keys" each produce a (uniform / geometric /
//! shifted-geometric) number of correlated edges (mimicking
//! `CompVFunc`'s Huffman multi-edge construction), then attempts to
//! peel it with a minimal 3-uniform peeler. Emits per-trial CSV rows
//! to stdout *as they finish* (not buffered), so a long sweep can be
//! piped into `tee` and inspected live.
//!
//! # Variants A vs B
//!
//! The fuse-filter formulas (`c`, `log2_seg_size`) are normally fed
//! the *actual* edge count of the graph being built. With
//! correlation, the same `s` keys produce `n_edges = s · w` edges,
//! and we want to know whether peelability is governed by `s` or by
//! `s · w`:
//!
//! * **Variant A**: feed `n_edges` into the formula. Vertex count is
//!   `c(n_edges) · n_edges` — the smaller `c` (since `c` decreases
//!   with input). Multi-edges are treated as fresh equations.
//! * **Variant B**: feed `s` (key count) into the formula. Vertex
//!   count is `c(s) · n_edges` — the larger `c`. Multi-edges are
//!   treated as perturbations on an `s`-key graph.
//!
//! If A peels reliably for all w → entropy-sharding is sound. If
//! only B works → multi-edges really are correlated noise and we
//! must shard by keys.
//!
//! # Distributions
//!
//! * `uniform`:     each key produces exactly `w` edges (worst case).
//! * `geom`:        each key produces `k + 1` edges with `k` drawn
//!                  from `Geometric(0.5)`. Mean ≈ 2, entropy ≈ 3.
//! * `shifted-geom`: each key produces `k + w` edges with `k` drawn
//!                   from `Geometric(0.5)`. Tunable lower bound.
//!
//! # Usage
//!
//! ```text
//! cargo run --release --example corr_peel_sweep -- \
//!     --variant a --shard-edge fuselge3 --distribution uniform \
//!     --n-list 10,100,1000,10000,100000,1000000,10000000,100000000 \
//!     --w-min 1 --w-max 16 --trials 5
//! ```

use clap::{Parser, ValueEnum};
use rand::distr::Distribution;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use rand_distr::Geometric;
use rayon::prelude::*;
use std::collections::HashMap;
use std::io::Write;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Copy, Clone, Debug, ValueEnum)]
enum ShardEdgeKind {
    /// Use the `FuseLge3*Shards`/`FuseLge3*NoShards` c & log2_seg_size formulas.
    Fuselge3,
    /// Use the `Fuse3*NoShards` (no-LGE) c & log2_seg_size formulas.
    Fuse3,
    /// Use `max(fuse3_log2_seg_size(n), fuselge3_log2_seg_size(n))`,
    /// i.e. never shrink segments below what plain Fuse3 picks but
    /// expand to FuseLge3's ε-cost formula when that's larger. Used
    /// only via `--seg-formula`; `c` stays from `--shard-edge`.
    Maxseg,
}

#[derive(Copy, Clone, Debug, ValueEnum, PartialEq, Eq)]
enum Variant {
    /// Both `c` and `log2_seg_size` keyed at edge count `s * w`.
    /// "Treat multi-edges as fresh equations." This is what
    /// CompVFunc currently does in production.
    A,
    /// Both `c` and `log2_seg_size` keyed at key count `s`.
    /// "Ignore multi-edges entirely." Tends to break because the
    /// segment count is sized for too few edges.
    B,
    /// `c` keyed at key count `s` but `log2_seg_size` keyed at
    /// edge count `s * w`. Isolates the *c-as-vertex-multiplier*
    /// effect from the segment-density effect: more vertices per
    /// edge (key-based c is larger) without distorting the band
    /// structure (edge-based segments are correctly sized).
    C,
}

#[derive(Copy, Clone, Debug, ValueEnum, PartialEq, Eq)]
enum DistKind {
    /// Each key contributes `w` edges (constant).
    Uniform,
    /// Each key contributes `Geometric(0.5) + 1` edges.
    Geom,
    /// Each key contributes `Geometric(0.5) + w` edges.
    ShiftedGeom,
    /// Each key contributes the Huffman codeword length for a value
    /// drawn from `Zipf(zipf_s, zipf_n)`. Simulates the realistic
    /// CompVFunc per-key edge count when values follow a Zipf law.
    Zipf,
}

#[derive(Parser, Debug)]
#[command(
    about = "Sweep 3-uniform correlated fuse-graph peelability",
    long_about = None
)]
struct Args {
    /// ShardEdge whose `c` formula to apply. The `log2_seg_size`
    /// formula comes from `--seg-formula` (default: same as `c`).​
    #[arg(long, value_enum, default_value_t = ShardEdgeKind::Fuselge3)]
    shard_edge: ShardEdgeKind,

    /// ShardEdge whose `log2_seg_size` formula to apply. Defaults to
    /// `--shard-edge`. Setting them differently lets us test
    /// hybrid configurations like "Fuse3 c with FuseLge3 segments".​
    #[arg(long, value_enum)]
    seg_formula: Option<ShardEdgeKind>,

    /// Variant `A` (formula sized at `s*w`) or `B` (formula sized at `s`).​
    #[arg(long, value_enum, default_value_t = Variant::A)]
    variant: Variant,

    /// Per-key edge-count distribution.​
    #[arg(long, value_enum, default_value_t = DistKind::Uniform)]
    distribution: DistKind,

    /// Number of trials per cell.​
    #[arg(short, long, default_value_t = 5)]
    trials: usize,

    /// Explicit comma-separated list of `s` (key count) values. If
    /// set, overrides `--s-max` / `--fixed-edges`. Example:
    /// `--n-list 10,100,1000,10000,100000,1000000,10000000,100000000`.​
    #[arg(long)]
    n_list: Option<String>,

    /// Minimum value of `s` to sweep (continuous range mode). When
    /// `--s-min` is set, sweeps every integer in `s_min..=s_max`.
    /// When unset, uses the sparse geometric progression from 2^16.​
    #[arg(long)]
    s_min: Option<usize>,

    /// Maximum value of `s` (number of groups/keys) to sweep, when
    /// `--n-list` is unset. If `--s-min` is also set, sweeps every
    /// integer in `s_min..=s_max`. Otherwise starts at `2^16` and
    /// multiplies by 1.5 until it exceeds this.​
    #[arg(long, default_value_t = 20_000_000)]
    s_max: usize,

    /// Maximum total edge count per cell. Cells exceeding this are
    /// skipped to control runtime and memory.​
    #[arg(long, default_value_t = 200_000_000)]
    edge_cap: usize,

    /// Minimum `w` (correlation width) to sweep. Used by `uniform`
    /// and `shifted-geom`. Ignored if `--w-list` is set.​
    #[arg(long, default_value_t = 1)]
    w_min: usize,

    /// Maximum `w` (correlation width) to sweep. Loop runs
    /// `w = w_min..=w_max`. Used by `uniform` and `shifted-geom`.
    /// Ignored by `geom` and `zipf`. Ignored if `--w-list` is set.​
    #[arg(long, default_value_t = 16)]
    w_max: usize,

    /// Explicit comma-separated list of `w` values. Overrides
    /// `--w-min` / `--w-max` when set. Useful for sparse jumps over
    /// a wide range (e.g. `1,2,4,8,16,32,57`).​
    #[arg(long)]
    w_list: Option<String>,

    /// Zipf exponent `s` for `--distribution zipf`. Values around
    /// 1.0 give heavy tails; values ≥ 2 are very skewed.​
    #[arg(long, default_value_t = 1.0)]
    zipf_s: f64,

    /// Zipf alphabet size `n` for `--distribution zipf`.​
    #[arg(long, default_value_t = 1000)]
    zipf_n: usize,

    /// If set, sweep only this single `c_delta` (instead of the
    /// default `{0, 0.01, 0.05, 0.10}` grid).​
    #[arg(long)]
    only_c_delta: Option<f64>,

    /// If set, sweep only this single `log2_seg_delta` (instead of
    /// the default `{-1, 0, +1}` grid).​
    #[arg(long)]
    only_seg_delta: Option<i32>,

    /// If set, hold `num_edges ≈ this value` and derive `s = N / w`
    /// for each `w`. Ignored when `--n-list` is set.​
    #[arg(long)]
    fixed_edges: Option<usize>,
}

// ── Inlined ShardEdge formulas ─────────────────────────────────────
//
// Two flavors:
//
// * `fuselge3_*`: piecewise-constant c, with LGE assumed for the
//   small (≤ 800K) and intermediate (≤ 20M) regimes.
// * `fuse3_*`:    continuous c (binary fuse filters paper), no LGE
//   anywhere.

const MAX_LIN_SIZE: usize = 800_000;
const MIN_FUSE_SHARD: usize = 10_000_000;
const A_LARGE: f64 = 0.41;
const B_LARGE: f64 = -3.0;

/// Baseline `c` for a graph of `n` edges, matching
/// `FuseLge3Shards::set_up_graphs`.
fn fuselge3_c(n: usize) -> f64 {
    if n <= 100 {
        1.23
    } else if n <= MAX_LIN_SIZE {
        1.125
    } else if n <= MIN_FUSE_SHARD / 2 {
        1.125
    } else if n <= MIN_FUSE_SHARD {
        1.12
    } else if n <= 2 * MIN_FUSE_SHARD {
        1.11
    } else {
        1.105
    }
}

/// Baseline `log2_seg_size` for a graph of `n` edges, matching
/// `FuseLge3Shards::set_up_graphs`.
fn fuselge3_log2_seg_size(n: usize) -> u32 {
    if n <= MAX_LIN_SIZE {
        (0.85 * (n.max(1) as f64).ln()).floor().max(1.0) as u32
    } else {
        let nf = (n.max(1) as f64).ln();
        let raw: f64 = if n <= 2 * MIN_FUSE_SHARD {
            nf / (3.33_f64).ln() + 2.25
        } else {
            A_LARGE * nf * nf.max(1.0).ln() + B_LARGE
        };
        raw.max(1.0) as u32
    }
}

/// Whether `FuseLge3Shards::set_up_graphs` would return `lge=true`.
fn fuselge3_expects_lge(n: usize) -> bool {
    n <= MAX_LIN_SIZE
}

/// Baseline `c` for a graph of `n` edges, matching `Fuse3NoShards`.
/// Smooth (binary fuse filters paper).
fn fuse3_c(n: usize) -> f64 {
    let n = n.max(2) as f64;
    0.875 + 0.25 * (1.0_f64).max((1e6_f64).ln() / n.ln())
}

/// Baseline `log2_seg_size` for a graph of `n` edges, matching
/// `Fuse3NoShards`.
fn fuse3_log2_seg_size(n: usize) -> u32 {
    let n = n.max(1) as f64;
    if (n as usize) <= 100_000_000 {
        (n.ln() / (3.33_f64).ln() + 2.25).floor() as u32
    } else {
        (A_LARGE * n.ln() * n.ln().max(1.0).ln() + B_LARGE).floor() as u32
    }
}

/// `Fuse3*` never uses LGE (it's the no-LGE variant).
fn fuse3_expects_lge(_n: usize) -> bool {
    false
}

/// Returns `(c, expects_lge)` from the c-formula chosen by
/// `c_kind` applied to `formula_input`. `Maxseg` is not a valid
/// c-formula (it only makes sense for segments).
fn baseline_c(c_kind: ShardEdgeKind, formula_input: usize) -> (f64, bool) {
    match c_kind {
        ShardEdgeKind::Fuselge3 => (
            fuselge3_c(formula_input),
            fuselge3_expects_lge(formula_input),
        ),
        ShardEdgeKind::Fuse3 => (fuse3_c(formula_input), fuse3_expects_lge(formula_input)),
        ShardEdgeKind::Maxseg => {
            panic!("--shard-edge maxseg has no c formula; use it only via --seg-formula")
        }
    }
}

/// Returns `log2_seg_size` from the seg-formula chosen by `seg_kind`
/// applied to `formula_input`.
fn baseline_log2_seg(seg_kind: ShardEdgeKind, formula_input: usize) -> u32 {
    match seg_kind {
        ShardEdgeKind::Fuselge3 => fuselge3_log2_seg_size(formula_input),
        ShardEdgeKind::Fuse3 => fuse3_log2_seg_size(formula_input),
        ShardEdgeKind::Maxseg => {
            fuse3_log2_seg_size(formula_input).max(fuselge3_log2_seg_size(formula_input))
        }
    }
}

// ── Zipf precomputation ────────────────────────────────────────────
//
// For `DistKind::Zipf` we precompute, once per run, two things:
//
// 1. A CDF table over ranks `1..=n`, used to inverse-transform-
//    sample a rank in O(log n).
// 2. The Huffman codeword length `len[rank]` for each rank, computed
//    by feeding a Zipf frequency table into `sux::func::codec::Huffman`.
//    This gives the per-key edge count corresponding to "encoding a
//    Zipf-distributed value with the same Huffman code CompVFunc
//    would build at construction time".

#[derive(Debug)]
struct ZipfData {
    /// `cdf[i] = P(rank ≤ i + 1)` for `i ∈ 0..n`.
    cdf: Vec<f64>,
    /// `lengths[rank]` for `rank ∈ 1..=n` (index 0 unused).
    lengths: Vec<u32>,
}

fn build_zipf(s: f64, n: usize) -> ZipfData {
    let h: f64 = (1..=n).map(|i| 1.0 / (i as f64).powf(s)).sum();
    let mut cdf = vec![0.0; n];
    let mut acc = 0.0;
    for i in 1..=n {
        acc += 1.0 / (i as f64).powf(s) / h;
        cdf[i - 1] = acc;
    }

    // Build a Huffman code over the Zipf frequencies and look up
    // the codeword length for each rank.
    use sux::func::codec::{Codec, Coder, Huffman};
    let scale: f64 = 1_000_000.0;
    let mut freqs: HashMap<u64, usize> = HashMap::new();
    for rank in 1..=n {
        let p = 1.0 / (rank as f64).powf(s) / h;
        let count = ((p * scale).round() as usize).max(1);
        freqs.insert(rank as u64, count);
    }
    let coder = <Huffman as Codec<u64>>::build_coder(&Huffman::new(), &freqs);
    let mut lengths = vec![0u32; n + 1];
    for rank in 1..=n {
        let len = coder.codeword_length(rank as u64);
        // Codeword length 0 happens only for the degenerate single-
        // symbol case; clamp to 1 so peels see at least one edge.
        lengths[rank] = len.max(1);
    }
    ZipfData { cdf, lengths }
}

fn sample_zipf_rank(cdf: &[f64], rng: &mut SmallRng) -> usize {
    // Inverse-CDF sampler. `partition_point` returns the smallest
    // index `i` for which `cdf[i] >= u`, which corresponds to rank
    // `i + 1`.
    let u: f64 = (rng.random::<u64>() >> 11) as f64 / ((1u64 << 53) as f64);
    cdf.partition_point(|&p| p < u) + 1
}

// ── Graph construction ──────────────────────────────────────────────

/// Derives `(num_vertices, l)` for a fuse-structured graph with
/// `num_edges` edges, density `c`, and segment size `2^log2_seg`.
/// `pad` is the maximum shift width (so the upper segment edge can
/// hold the highest-shift vertex).
fn fuse_dims(num_edges: usize, c: f64, log2_seg: u32, pad: usize) -> (usize, usize) {
    let seg_size = 1usize << log2_seg;
    let target_vertices = (c * num_edges as f64).ceil() as usize;
    let l = target_vertices.div_ceil(seg_size).saturating_sub(2).max(1);
    let num_vertices = (l + 2) * seg_size + pad;
    (num_vertices, l)
}

/// Generates a fuse-structured 3-uniform hypergraph with CompVFunc-
/// style correlation. For each key, we sample a per-key edge count
/// `w_k` from the distribution, then emit `w_k` edges shifted by
/// `(w_k − 1 − l_shift)` for `l_shift ∈ 0..w_k`. Returns the edges,
/// the total edge count, and the largest per-key `w` actually seen.
fn gen_correlated_graph(
    s: usize,
    dist: DistKind,
    w_param: usize,
    zipf: Option<&ZipfData>,
    l: usize,
    log2_seg: u32,
    seed: u64,
) -> (Vec<[u32; 3]>, usize, usize) {
    let seg_size = 1u64 << log2_seg;
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut edges: Vec<[u32; 3]> = Vec::with_capacity(s * w_param.max(2));
    let geo = Geometric::new(0.5).unwrap();
    let mut max_w_seen = 0usize;
    for _ in 0..s {
        let w_k: usize = match dist {
            DistKind::Uniform => w_param,
            DistKind::Geom => (geo.sample(&mut rng) as usize) + 1,
            DistKind::ShiftedGeom => (geo.sample(&mut rng) as usize) + w_param,
            DistKind::Zipf => {
                let z = zipf.expect("zipf distribution requires precomputed ZipfData");
                let rank = sample_zipf_rank(&z.cdf, &mut rng);
                z.lengths[rank] as usize
            }
        };
        if w_k == 0 {
            continue;
        }
        max_w_seen = max_w_seen.max(w_k);
        let s_idx: u64 = rng.random_range(0..l as u64);
        let o0: u64 = rng.random_range(0..seg_size);
        let o1: u64 = rng.random_range(0..seg_size);
        let o2: u64 = rng.random_range(0..seg_size);
        let b0 = s_idx * seg_size + o0;
        let b1 = (s_idx + 1) * seg_size + o1;
        let b2 = (s_idx + 2) * seg_size + o2;
        for l_shift in 0..w_k as u64 {
            let off = (w_k as u64 - 1) - l_shift;
            edges.push([(b0 + off) as u32, (b1 + off) as u32, (b2 + off) as u32]);
        }
    }
    let total = edges.len();
    (edges, total, max_w_seen)
}

/// Fast pre-pass: sample the per-key edge counts only, so we can
/// decide `(c, log2_seg, num_vertices)` *before* generating the
/// graph. Re-seeded identically when actually generating, so the
/// per-key counts match.
fn estimate_total_edges(
    s: usize,
    dist: DistKind,
    w_param: usize,
    zipf: Option<&ZipfData>,
    seed: u64,
) -> (usize, usize) {
    if matches!(dist, DistKind::Uniform) {
        return (s * w_param, w_param);
    }
    let mut rng = SmallRng::seed_from_u64(seed);
    let geo = Geometric::new(0.5).unwrap();
    let mut total = 0usize;
    let mut max_w_seen = 0usize;
    for _ in 0..s {
        let w_k = match dist {
            DistKind::Uniform => w_param,
            DistKind::Geom => (geo.sample(&mut rng) as usize) + 1,
            DistKind::ShiftedGeom => (geo.sample(&mut rng) as usize) + w_param,
            DistKind::Zipf => {
                let z = zipf.expect("zipf requires precomputed ZipfData");
                let rank = sample_zipf_rank(&z.cdf, &mut rng);
                z.lengths[rank] as usize
            }
        };
        max_w_seen = max_w_seen.max(w_k);
        total += w_k;
    }
    (total, max_w_seen)
}

// ── Minimal inline peeler ──────────────────────────────────────────

/// Attempts to peel `edges` using a simple 3-uniform XOR-based peel.
/// Returns the number of edges peeled (== `edges.len()` iff success).
fn peel_simple(num_vertices: usize, edges: &[[u32; 3]]) -> usize {
    let mut edge_xor = vec![0u32; num_vertices];
    let mut degree = vec![0u8; num_vertices];

    for (i, &[a, b, c]) in edges.iter().enumerate() {
        let stored = (i + 1) as u32;
        edge_xor[a as usize] ^= stored;
        edge_xor[b as usize] ^= stored;
        edge_xor[c as usize] ^= stored;
        let (da, oa) = degree[a as usize].overflowing_add(1);
        let (db, ob) = degree[b as usize].overflowing_add(1);
        let (dc, oc) = degree[c as usize].overflowing_add(1);
        if oa || ob || oc {
            return 0;
        }
        degree[a as usize] = da;
        degree[b as usize] = db;
        degree[c as usize] = dc;
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
        let edge_idx = (stored - 1) as usize;
        n_peeled += 1;
        let [a, b, c] = edges[edge_idx];
        for &u in &[a, b, c] {
            let uu = u as usize;
            degree[uu] -= 1;
            edge_xor[uu] ^= stored;
            if degree[uu] == 1 {
                stack.push(u);
            }
        }
    }

    n_peeled
}

// ── Sweep driver ───────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Cell {
    s: usize,
    /// Distribution parameter (`w` for uniform/shifted-geom; unused
    /// for geom but kept for cell identity).
    w_param: usize,
    log2_seg_delta: i32,
    c_delta: f64,
    trial: usize,
}

#[derive(Debug, Clone)]
struct TrialResult {
    cell: Cell,
    num_edges: usize,
    max_w_seen: usize,
    num_vertices: usize,
    log2_seg: u32,
    c: f64,
    formula_input: usize,
    seed: u64,
    n_peeled: usize,
    expected_lge: bool,
}

fn run_trial(args: &Args, zipf: Option<&ZipfData>, cell: &Cell) -> TrialResult {
    let Cell {
        s,
        w_param,
        log2_seg_delta,
        c_delta,
        trial,
    } = *cell;

    let seed = hash_seed(s, w_param, log2_seg_delta, c_delta, trial);

    // First, estimate total edges (and max per-key w) so we can size
    // the formula input correctly.
    let (num_edges_est, _max_w_est) =
        estimate_total_edges(s, args.distribution, w_param, zipf, seed);

    // For Variants A and B, both c and log2_seg are keyed at the
    // same input. For Variant C, c is keyed at `s` (key count) but
    // log2_seg is keyed at `num_edges` (edge count).
    let (formula_input_c, formula_input_seg) = match args.variant {
        Variant::A => (num_edges_est, num_edges_est),
        Variant::B => (s, s),
        Variant::C => (s, num_edges_est),
    };
    let formula_input = formula_input_c; // recorded for the CSV

    // c comes from `--shard-edge`, log2_seg from `--seg-formula`
    // (defaults to `--shard-edge`). Letting these differ lets us
    // probe hybrid configurations like "Fuse3 c with FuseLge3 seg".
    let seg_kind = args.seg_formula.unwrap_or(args.shard_edge);
    let (base_c, expected_lge) = baseline_c(args.shard_edge, formula_input_c);
    let base_log2_seg = baseline_log2_seg(seg_kind, formula_input_seg);
    let c = base_c + c_delta;
    let log2_seg = ((base_log2_seg as i32) + log2_seg_delta).max(1) as u32;

    // Pad needs to cover the maximum per-key edge count we might
    // see. For uniform that's just `w_param`; for geom/shifted we
    // use a generous upper bound (w_param + 64) and verify after
    // generation.
    let pad = match args.distribution {
        DistKind::Uniform => w_param,
        DistKind::Geom => 1 + 64,
        DistKind::ShiftedGeom => w_param + 64,
        DistKind::Zipf => zipf
            .map(|z| *z.lengths.iter().max().unwrap_or(&1) as usize)
            .unwrap_or(64),
    };
    let (num_vertices, l) = fuse_dims(num_edges_est, c, log2_seg, pad);

    // Now actually generate the graph. Re-seeded identically so the
    // per-key counts match the estimate.
    let (edges, num_edges, max_w_seen) =
        gen_correlated_graph(s, args.distribution, w_param, zipf, l, log2_seg, seed);

    // Sanity: if the actual max-w exceeds our pad, the upper-segment
    // vertices may overflow — bail with n_peeled = 0 to flag this
    // cell. Realistically the geom tail should never reach 64, but
    // we mark such cases explicitly.
    if max_w_seen > pad {
        return TrialResult {
            cell: cell.clone(),
            num_edges,
            max_w_seen,
            num_vertices,
            log2_seg,
            c,
            formula_input,
            seed,
            n_peeled: 0,
            expected_lge,
        };
    }

    let n_peeled = peel_simple(num_vertices, &edges);

    TrialResult {
        cell: cell.clone(),
        num_edges,
        max_w_seen,
        num_vertices,
        log2_seg,
        c,
        formula_input,
        seed,
        n_peeled,
        expected_lge,
    }
}

fn hash_seed(s: usize, w: usize, seg_delta: i32, c_delta: f64, trial: usize) -> u64 {
    let c_delta_bits = (c_delta * 1e6).round() as u64;
    let mut x: u64 = 0xcbf29ce484222325;
    for v in [
        s as u64,
        w as u64,
        (seg_delta as i64) as u64,
        c_delta_bits,
        trial as u64,
    ] {
        x ^= v;
        x = x.wrapping_mul(0x100000001b3);
    }
    x
}

fn parse_n_list(s: &str) -> Vec<usize> {
    s.split(',')
        .map(|tok| {
            tok.trim()
                .parse()
                .expect("--n-list must be comma-separated integers")
        })
        .collect()
}

fn build_cells(args: &Args) -> Vec<Cell> {
    let seg_deltas: Vec<i32> = match args.only_seg_delta {
        Some(d) => vec![d],
        None => vec![-1, 0, 1],
    };
    let c_deltas: Vec<f64> = match args.only_c_delta {
        Some(d) => vec![d],
        None => vec![0.0, 0.01, 0.05, 0.10],
    };

    // For `geom` and `zipf` we don't sweep `w_param`. For `uniform`
    // and `shifted-geom`, we sweep either `--w-list` (if set) or
    // `w_min..=w_max`.
    let w_params: Vec<usize> = match args.distribution {
        DistKind::Geom | DistKind::Zipf => vec![0],
        DistKind::Uniform | DistKind::ShiftedGeom => match &args.w_list {
            Some(s) => s
                .split(',')
                .map(|t| {
                    t.trim()
                        .parse()
                        .expect("--w-list must be comma-separated integers")
                })
                .collect(),
            None => (args.w_min..=args.w_max).collect(),
        },
    };

    let mut cells: Vec<Cell> = Vec::new();

    let s_values: Vec<usize> = if let Some(n_list) = &args.n_list {
        parse_n_list(n_list)
    } else if let Some(target_edges) = args.fixed_edges {
        // Per-w derived list, replicated across w in the inner loop
        // — handled below via single sentinel (we'll override in the
        // per-w branch).
        // To preserve the old fixed_edges semantics, build cells
        // directly here and return early.
        for &w in &w_params {
            if w == 0 {
                continue;
            }
            let s = target_edges / w;
            if s == 0 {
                continue;
            }
            for &seg_delta in &seg_deltas {
                for &c_delta in &c_deltas {
                    for trial in 0..args.trials {
                        cells.push(Cell {
                            s,
                            w_param: w,
                            log2_seg_delta: seg_delta,
                            c_delta,
                            trial,
                        });
                    }
                }
            }
        }
        return cells;
    } else if let Some(s_min) = args.s_min {
        (s_min..=args.s_max).collect()
    } else {
        let mut s: usize = 1 << 16;
        let mut out = Vec::new();
        while s <= args.s_max {
            out.push(s);
            let next = (s as f64 * 1.5).ceil() as usize;
            if next == s {
                break;
            }
            s = next;
        }
        out
    };

    for &s in &s_values {
        for &w in &w_params {
            // Quick-reject cells that obviously exceed the edge cap.
            // For uniform: exact. For geom/shifted-geom: upper bound
            // = s * (w + 32) — generous enough for the tail.
            let edge_estimate = match args.distribution {
                DistKind::Uniform => s.saturating_mul(w),
                DistKind::Geom => s.saturating_mul(1 + 8),
                DistKind::ShiftedGeom => s.saturating_mul(w + 8),
                // For Zipf we conservatively budget ~2× the expected
                // mean codeword length (≈ entropy of the chosen
                // Zipf). Using 16 as a generous upper bound for any
                // alphabet ≤ 1000 with s ≥ 1.
                DistKind::Zipf => s.saturating_mul(16),
            };
            if edge_estimate > args.edge_cap {
                continue;
            }
            for &seg_delta in &seg_deltas {
                for &c_delta in &c_deltas {
                    for trial in 0..args.trials {
                        cells.push(Cell {
                            s,
                            w_param: w,
                            log2_seg_delta: seg_delta,
                            c_delta,
                            trial,
                        });
                    }
                }
            }
        }
    }
    cells
}

fn main() {
    let args = Args::parse();

    eprintln!(
        "corr_peel_sweep: shard_edge={:?} variant={:?} dist={:?} trials={}",
        args.shard_edge, args.variant, args.distribution, args.trials
    );

    let zipf_data: Option<ZipfData> = if matches!(args.distribution, DistKind::Zipf) {
        let z = build_zipf(args.zipf_s, args.zipf_n);
        let max_len = z.lengths.iter().copied().max().unwrap_or(0);
        let mean_len: f64 =
            z.lengths.iter().skip(1).copied().sum::<u32>() as f64 / args.zipf_n as f64;
        eprintln!(
            "  zipf: s={} n={} mean_codeword_length={:.3} max_codeword_length={}",
            args.zipf_s, args.zipf_n, mean_len, max_len
        );
        Some(z)
    } else {
        None
    };
    let zipf_ref = zipf_data.as_ref();

    let cells = build_cells(&args);
    let total = cells.len();
    eprintln!("Total trials to run: {total}");

    // CSV header — emitted once, before any worker writes.
    let stdout = std::io::stdout();
    {
        let mut out = stdout.lock();
        writeln!(
            out,
            "shard_edge,variant,dist,s,w_param,log2_seg_delta,c_delta,formula_input,num_edges,max_w_seen,num_vertices,log2_seg,c,trial,seed,n_peeled,peel_frac,success,expected_lge"
        )
        .unwrap();
        out.flush().unwrap();
    }

    // Mutex around stdout for incremental output. `parking_lot`
    // would be faster but `std::sync::Mutex` is fine — we lock
    // briefly per row.
    let stdout_mutex = Mutex::new(std::io::stdout());

    let done = AtomicUsize::new(0);
    let start = std::time::Instant::now();

    cells.par_iter().for_each(|cell| {
        let r = run_trial(&args, zipf_ref, cell);

        // Format the row.
        let success = r.n_peeled == r.num_edges && r.num_edges > 0;
        let peel_frac = if r.num_edges == 0 {
            0.0
        } else {
            r.n_peeled as f64 / r.num_edges as f64
        };
        let row = format!(
            "{:?},{:?},{:?},{},{},{},{:.4},{},{},{},{},{},{:.4},{},{:#x},{},{:.4},{},{}",
            args.shard_edge,
            args.variant,
            args.distribution,
            r.cell.s,
            r.cell.w_param,
            r.cell.log2_seg_delta,
            r.cell.c_delta,
            r.formula_input,
            r.num_edges,
            r.max_w_seen,
            r.num_vertices,
            r.log2_seg,
            r.c,
            r.cell.trial,
            r.seed,
            r.n_peeled,
            peel_frac,
            if success { 1 } else { 0 },
            if r.expected_lge { 1 } else { 0 }
        );

        // Emit incrementally.
        {
            let mut out = stdout_mutex.lock().unwrap();
            writeln!(out, "{row}").unwrap();
            out.flush().unwrap();
        }

        let count = done.fetch_add(1, Ordering::Relaxed) + 1;
        if count % 100 == 0 || count == total {
            let elapsed = start.elapsed().as_secs_f64();
            let rate = count as f64 / elapsed;
            let eta = (total - count) as f64 / rate;
            let _ = writeln!(
                std::io::stderr(),
                "[{:>6}/{:<6}] {:.1}% elapsed={:.1}s rate={:.0}/s eta={:.1}s",
                count,
                total,
                (count as f64 / total as f64) * 100.0,
                elapsed,
                rate,
                eta
            );
        }
    });

    let total_time = start.elapsed().as_secs_f64();
    eprintln!("Done. Total runtime: {:.1}s", total_time);
}
