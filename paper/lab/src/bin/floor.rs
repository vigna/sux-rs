//! Window floor and self-collisions of the first level of PHast-R (Sections
//! 3 and 6 of the paper), on the same keys and hashes as `cmp`.
//!
//! The *floor* is the minimum number of keys that every placement of the
//! first level must bump given only the slice starts of the keys (each key
//! can be placed only in the L slots of its slice, and there are as many
//! slots as keys): since all slices have the same length, it is attained by
//! placing keys in order of slice start, each in the first free slot of its
//! slice (earliest deadline first). The tool also prints the bound D − L + 1
//! of Proposition 1 and the estimate n/(2L) of the paper.
//!
//! With `--layouts`, it builds PHast-R single-threaded with the given numbers
//! of layouts (as base-2 logarithms) and reports, for the buckets of the first
//! level, how many have a self-collision (two keys with the same base slot) in
//! every layout, and how many of the bumped buckets do.
//!
//! Usage: floor [-n keys] [-s seed bits] [--log2-slice-len l] [--lambda λ]
//! [--layouts 0,1,2]

use clap::Parser;
use dsi_progress_logger::no_logging;
use lab::GxKey;
use sux::func::{PHastR, PHastRBuilder};
use sux::utils::ToSig;

#[derive(Parser)]
struct Args {
    /// The number of keys.
    #[arg(short, default_value_t = 10_000_000)]
    n: usize,
    /// The number of bits of a seed.
    #[arg(short, default_value_t = 8)]
    s: u32,
    /// The base-2 logarithm of the maximum slice length.
    #[arg(long, default_value_t = 10)]
    log2_slice_len: u32,
    /// The expected bucket size.
    #[arg(long, default_value_t = 4.25)]
    lambda: f64,
    /// Comma-separated base-2 logarithms of the numbers of layouts of the
    /// structures to analyze (none by default).
    #[arg(long, value_delimiter = ',')]
    layouts: Vec<u32>,
    /// Generates a different key set.
    #[arg(long, default_value_t = 0)]
    key_seed: u64,
}

#[inline(always)]
fn mul_hi(a: u64, b: u64) -> u64 {
    ((a as u128 * b as u128) >> 64) as u64
}

/// Prints D − W + 1, the minimum number of bumped keys and n / (2W), given
/// the number of keys whose slice starts at each slot (the output range has
/// as many slots as keys).
fn floor(starts: &[u8], w: usize) {
    let n = starts.len();
    // The range D of X(t) = t − #{x | σ_x < t}
    let (mut below, mut min, mut max) = (0i64, 0i64, 0i64);
    for (t, &c) in starts.iter().enumerate() {
        below += c as i64;
        let x = t as i64 + 1 - below;
        (min, max) = (min.min(x), max.max(x));
    }
    // Earliest deadline first: waiting[i % w] is the number of waiting keys
    // whose slice starts at i, for i in (t − w, t]
    let (mut waiting, mut queued, mut bumped) = (vec![0u64; w], 0u64, 0u64);
    let mut oldest = 0;
    for t in 0..n {
        if t >= w {
            // Keys whose slice is [t − w, t) can no longer be placed
            let expired = std::mem::take(&mut waiting[(t - w) % w]);
            bumped += expired;
            queued -= expired;
        }
        waiting[t % w] = starts[t] as u64;
        queued += starts[t] as u64;
        if queued > 0 {
            oldest = oldest.max((t + 1).saturating_sub(w));
            while waiting[oldest % w] == 0 {
                oldest += 1;
            }
            waiting[oldest % w] -= 1;
            queued -= 1;
        }
    }
    bumped += queued;
    println!(
        "n={n} W={w}: floor {bumped} ({:.4}%), D-W+1 {} ({:.4}%), n/(2W) {:.0} ({:.4}%)",
        100.0 * bumped as f64 / n as f64,
        max - min - w as i64 + 1,
        100.0 * (max - min - w as i64 + 1) as f64 / n as f64,
        n as f64 / (2.0 * w as f64),
        100.0 / (2.0 * w as f64)
    );
}

fn main() {
    let a = Args::parse();
    let n = a.n;
    let keys: Vec<GxKey> = (0..n as u64)
        .map(|i| GxKey((i + a.key_seed.wrapping_mul(1 << 40)).wrapping_mul(0x9e3779b97f4a7c15) ^ 0x1234567))
        .collect();
    // The geometry of the first level (see PHastRBuilder::geometry)
    let l = (n / 2 + 1).next_power_of_two().min(1 << a.log2_slice_len);
    let num_slices = (n + 1 - l) as u64;
    let mut starts = vec![0u8; n];
    for &k in &keys {
        let s = &mut starts[mul_hi(GxKey::to_sig(k, 0)[0], num_slices) as usize];
        *s = s.checked_add(1).expect("too many keys with the same slice start");
    }
    floor(&starts, l);
    drop(starts);

    for &log2_layouts in &a.layouts {
        let b = PHastRBuilder::default()
            .seed_bits(a.s)
            .log2_slice_len(a.log2_slice_len)
            .bucket_size(a.lambda)
            .log2_layouts(log2_layouts);
        let f: PHastR<GxKey, Box<[u8]>> = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| PHastR::try_par_new_with_builder(&keys, b, no_logging![]).unwrap());
        let buckets = (n as f64 / a.lambda).round().max(1.0) as u64 | 1;
        let scale = l.ilog2().saturating_sub(a.s);
        let field = 64 >> log2_layouts;
        // The base slot of a key in layout r is its slot for seed r
        let base = |h: u64, r: u64| {
            let o = h.wrapping_mul(buckets);
            let offset = (o >> (r as u32 * field)).wrapping_add(r << scale);
            mul_hi(h, num_slices) + (offset & (l as u64 - 1))
        };
        let mut hb: Vec<(u64, u64, bool)> = keys
            .iter()
            .map(|&k| {
                let h = GxKey::to_sig(k, 0)[0];
                (mul_hi(h, buckets), h, f.is_bumped(k))
            })
            .collect();
        hb.sort_unstable_by_key(|x| (x.0, x.1));
        let layouts = 1u64 << log2_layouts;
        let (mut nonempty, mut sc, mut sc_keys) = (0usize, 0usize, 0usize);
        let (mut bumped, mut bumped_keys, mut sc_bumped, mut sc_bumped_keys) = (0, 0, 0, 0);
        let mut slots = vec![];
        for bucket in hb.chunk_by(|x, y| x.0 == y.0) {
            let k = bucket.len();
            nonempty += 1;
            // Checks that the buckets computed here are those of the structure
            assert!(bucket.iter().all(|x| x.2 == bucket[0].2));
            // A self-collision in every layout
            let all = (0..layouts).all(|r| {
                slots.clear();
                slots.extend(bucket.iter().map(|x| base(x.1, r)));
                slots.sort_unstable();
                slots.windows(2).any(|w| w[0] == w[1])
            });
            if all {
                sc += 1;
                sc_keys += k;
            }
            if bucket[0].2 {
                bumped += 1;
                bumped_keys += k;
                if all {
                    sc_bumped += 1;
                    sc_bumped_keys += k;
                }
            }
        }
        let lam = a.lambda;
        println!(
            "S={} L={l} lambda={lam} R={layouts}: bumped {:.3}% of the keys ({:.3}% of the nonempty buckets); \
             self-collision in every layout: {:.4}% of the nonempty buckets ({:.4}% of the keys), \
             {:.2}% of the bumped buckets ({:.2}% of the bumped keys); \
             first-order estimate for one layout: {:.4}% of the keys",
            a.s,
            100.0 * bumped_keys as f64 / n as f64,
            100.0 * bumped as f64 / nonempty as f64,
            100.0 * sc as f64 / nonempty as f64,
            100.0 * sc_keys as f64 / n as f64,
            100.0 * sc_bumped as f64 / bumped.max(1) as f64,
            100.0 * sc_bumped_keys as f64 / bumped_keys.max(1) as f64,
            100.0 * lam * (lam + 2.0) / (2.0 * l as f64)
        );
    }
}
