//! Anatomy of the space of PHast+, with or without wrapping (Section 2 of the
//! paper), measured on the reference implementation `ph` (using the hidden
//! accessors `Function2::level0_conf`, `level0_seed`, and `component_sizes`).
//!
//! Reports the space breakdown, the fraction of keys bumped from the first
//! level and its dependence on bucket size, the buckets whose keys have
//! coinciding base positions (self-collisions, which no shift can resolve),
//! and the empirical entropy of the seeds of the first level.
//!
//! Usage: anatomy [-n keys] [-s seed bits] [-l lambda] [-w multiplier]

use clap::Parser;
use ph::phast::{
    Conf, Core, DefaultCompressedArray, Function2, GenericCore, SeedChooserConf, SeedChooserCore,
    ShiftOnly, ShiftOnlyWrapped,
};
use ph::seedable_hash::BuildGxHash;
use ph::seeds::{Bits8, BitsFast, SeedSize};
use ph::{BuildSeededHasher, GetSize};

#[derive(Parser)]
struct Args {
    #[arg(short, default_value_t = 10_000_000)]
    n: usize,
    #[arg(short, default_value_t = 8)]
    s: u8,
    #[arg(short, default_value_t = 5.25)]
    l: f64,
    /// The multiplier of PHast+ with wrapping (0 for PHast+ without
    /// wrapping).
    #[arg(short, default_value_t = 0)]
    w: u8,
    /// Generates a different key set.
    #[arg(long, default_value_t = 0)]
    key_seed: u64,
}

fn analyze<SS: SeedSize, SC: SeedChooserConf>(keys: &[u64], ss: SS, sc: SC, a: &Args) {
    let n = keys.len();
    let core = sc.core();
    let conf = Conf::generic_with_hash(ss, (a.l * 100.0).round() as u32, BuildGxHash);
    let f: Function2<GenericCore, SS, SC::Core, DefaultCompressedArray, BuildGxHash> =
        Function2::with_slice_conf_sc(keys, conf, sc);
    let bits = |bytes: usize| bytes as f64 * 8.0 / n as f64;
    let total = bits(f.size_bytes());
    let (l0, remap, further) = f.component_sizes();
    let conf = *f.level0_conf();
    println!(
        "PHast+ S={} lambda={} wrap={} L={} n={n}: {total:.4} bits/key \
         (first level {:.4}, remapping {:.4}, further levels {:.4}, other {:.4})",
        a.s,
        a.l,
        a.w,
        conf.slice_len(),
        bits(l0),
        bits(remap),
        bits(further),
        total - bits(l0) - bits(remap) - bits(further)
    );

    // Hashes of the first level, grouped by bucket
    let mut hb: Vec<(usize, u64)> = keys
        .iter()
        .map(|k| {
            let h = BuildGxHash.hash_one(k, 0);
            (conf.bucket_for(h), h)
        })
        .collect();
    hb.sort_unstable();

    let nb = conf.buckets_num();
    let max_size = 64;
    let mut buckets_by_size = vec![0usize; max_size + 1];
    let mut bumped_by_size = vec![0usize; max_size + 1];
    let (mut bumped_keys, mut bumped_buckets) = (0usize, 0usize);
    let (mut sc_buckets, mut sc_bumped_buckets, mut sc_keys) = (0usize, 0usize, 0usize);
    let mut seed_count = vec![0usize; 1 << a.s];
    let mut i = 0;
    let mut bases = vec![];
    for b in 0..nb {
        let start = i;
        while i < hb.len() && hb[i].0 == b {
            i += 1;
        }
        let k = i - start;
        let seed = f.level0_seed(b);
        seed_count[seed as usize] += 1;
        if k == 0 {
            continue;
        }
        buckets_by_size[k.min(max_size)] += 1;
        // A bucket self-collides if two keys have the same position for
        // the first seed (and thus for all seeds, or, with wrapping, for all
        // seeds but those for which exactly one of the two has wrapped)
        bases.clear();
        bases.extend(hb[start..i].iter().map(|&(_, h)| core.f(h, 1, &conf)));
        bases.sort_unstable();
        let sc = bases.windows(2).any(|w| w[0] == w[1]);
        if sc {
            sc_buckets += 1;
            sc_keys += k;
        }
        if seed == 0 {
            bumped_buckets += 1;
            bumped_keys += k;
            bumped_by_size[k.min(max_size)] += 1;
            if sc {
                sc_bumped_buckets += 1;
            }
        }
    }
    let nonempty: usize = buckets_by_size.iter().sum();
    println!(
        "bumped keys {:.3}% (beta), bumped buckets {:.3}% of the nonempty ones",
        100.0 * bumped_keys as f64 / n as f64,
        100.0 * bumped_buckets as f64 / nonempty as f64
    );
    println!(
        "self-colliding buckets {:.3}% of the nonempty ones, {:.1}% of the bumped ones \
         (containing {:.3}% of the keys)",
        100.0 * sc_buckets as f64 / nonempty as f64,
        100.0 * sc_bumped_buckets as f64 / bumped_buckets.max(1) as f64,
        100.0 * sc_keys as f64 / n as f64
    );
    let entropy = |counts: &[usize]| {
        let tot: usize = counts.iter().sum();
        counts
            .iter()
            .filter(|&&c| c > 0)
            .map(|&c| {
                let p = c as f64 / tot as f64;
                -p * p.log2()
            })
            .sum::<f64>()
    };
    println!(
        "empirical entropy of the first-level seeds: {:.3} bits ({:.3} bits excluding bumped buckets)",
        entropy(&seed_count),
        entropy(&seed_count[1..])
    );
    println!("size: buckets bumped rate");
    for (k, (&nbk, &bk)) in buckets_by_size.iter().zip(&bumped_by_size).enumerate() {
        if nbk > 0 {
            println!("{k:4}: {nbk:10} {bk:10} {:.4}", bk as f64 / nbk as f64);
        }
    }
}

fn main() {
    let a = Args::parse();
    let keys: Vec<u64> = (0..a.n as u64)
        .map(|i| {
            (i + a.key_seed.wrapping_mul(1 << 40)).wrapping_mul(0x9e3779b97f4a7c15) ^ 0x1234567
        })
        .collect();
    match (a.s, a.w) {
        (8, 0) => analyze(&keys, Bits8, ShiftOnly, &a),
        (8, 1) => analyze(&keys, Bits8, ShiftOnlyWrapped::<1>, &a),
        (8, 2) => analyze(&keys, Bits8, ShiftOnlyWrapped::<2>, &a),
        (8, 3) => analyze(&keys, Bits8, ShiftOnlyWrapped::<3>, &a),
        (s, 0) => analyze(&keys, BitsFast(s), ShiftOnly, &a),
        (s, 1) => analyze(&keys, BitsFast(s), ShiftOnlyWrapped::<1>, &a),
        (s, 2) => analyze(&keys, BitsFast(s), ShiftOnlyWrapped::<2>, &a),
        (s, 3) => analyze(&keys, BitsFast(s), ShiftOnlyWrapped::<3>, &a),
        _ => panic!("unsupported multiplier"),
    }
}
