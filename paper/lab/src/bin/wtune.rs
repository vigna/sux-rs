//! Tuning of the size-dependent priority weights of the buckets of PHast-R.
//!
//! The objective is the mean space in bits per key of single-threaded
//! constructions on a few key sets, which are built concurrently. Space does
//! not depend on the hardware, so tuning can be run anywhere.
//!
//! Unless starting weights are given, the search starts from the best weights
//! of the form w(1) = -d and w(k) = a ln k for k > 1 on a grid of values of a
//! and d. Then, it refines the weights by coordinate descent with decreasing
//! steps, moving either a single weight or all the weights from a given size
//! on (i.e., a single difference between consecutive weights). With --eval,
//! the given weights (or the default ones, if none are given) are just
//! evaluated: use a different --key-seed to evaluate them on key sets
//! different from those used for tuning.
//!
//! Usage: wtune [options] <seed bits> <log2 slice len> <bucket size>
use clap::Parser;
use lab::phast_r_bits_per_key;
use sux::func::PHastRBuilder;

#[derive(Parser)]
struct Args {
    /// The number of bits per seed.
    seed_bits: u32,
    /// The base-2 logarithm of the slice length.
    log2_slice_len: u32,
    /// The expected bucket size.
    bucket_size: f64,
    /// The number of keys of each key set.
    #[arg(short, long, default_value_t = 10_000_000)]
    n: u64,
    /// The base-2 logarithm of the number of layouts.
    #[arg(long, default_value_t = 2)]
    log2_layouts: u32,
    /// The number of key sets.
    #[arg(long, default_value_t = 8)]
    sets: u64,
    /// The seed of the key sets.
    #[arg(long, default_value_t = 0)]
    key_seed: u64,
    /// Starting weights (seven comma-separated values).
    #[arg(short, long, value_delimiter = ',', allow_hyphen_values = true)]
    weights: Option<Vec<i64>>,
    /// Just evaluates the given weights.
    #[arg(long)]
    eval: bool,
}

/// Returns the mean space in bits per key of single-threaded constructions on
/// the given key sets.
fn eval(keysets: &[Vec<u64>], b: &PHastRBuilder, s: u32) -> f64 {
    let tot: f64 = std::thread::scope(|scope| {
        let handles: Vec<_> = keysets
            .iter()
            .map(|keys| {
                scope.spawn(move || {
                    // A pool with one thread yields the structure of a
                    // single-threaded construction
                    rayon::ThreadPoolBuilder::new()
                        .num_threads(1)
                        .build()
                        .unwrap()
                        .install(|| phast_r_bits_per_key(keys, b, s))
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).sum()
    });
    tot / keysets.len() as f64
}

fn main() {
    let a = Args::parse();
    let s = a.seed_bits;
    let keysets: Vec<Vec<u64>> = (0..a.sets)
        .map(|k| {
            let base = ((a.key_seed << 20) + k) << 40;
            (0..a.n)
                .map(|i| (base + i).wrapping_mul(0x9e3779b97f4a7c15) ^ 0xabcdef)
                .collect()
        })
        .collect();
    let base = PHastRBuilder::default()
        .seed_bits(s)
        .log2_slice_len(a.log2_slice_len)
        .bucket_size(a.bucket_size)
        .log2_layouts(a.log2_layouts);
    let f = |w: [i64; 7]| eval(&keysets, &base.clone().weights(w), s);
    let given = a
        .weights
        .map(|w| <[i64; 7]>::try_from(w).expect("seven weights are needed"));

    if a.eval {
        match given {
            Some(w) => println!("eval {:.5} {w:?}", f(w)),
            None => println!("eval {:.5} default weights", eval(&keysets, &base, s)),
        }
        return;
    }

    let (mut w, mut best) = match given {
        Some(w) => (w, f(w)),
        None => {
            // Grid search on w(1) = -d, w(k) = a ln k
            let mut best = (f64::INFINITY, [0; 7]);
            for a in (40_000..=200_000).step_by(20_000) {
                for d in (-50_000..=150_000).step_by(50_000) {
                    let w: [i64; 7] = std::array::from_fn(|i| {
                        if i == 0 {
                            -d
                        } else {
                            (a as f64 * ((i + 1) as f64).ln()).round() as i64
                        }
                    });
                    let v = f(w);
                    println!("grid a={a} d={d}: {v:.5}");
                    if v < best.0 {
                        best = (v, w);
                    }
                }
            }
            (best.1, best.0)
        }
    };
    println!("start {best:.5} {w:?}");

    let mut step = 32_000i64;
    while step >= 1000 {
        let mut improved = true;
        while improved {
            improved = false;
            // Moves of a single weight, and of all the weights from size
            // i + 1 on
            for (i, suffix) in (0..7).map(|i| (i, false)).chain((1..7).map(|i| (i, true))) {
                for dir in [-1i64, 1] {
                    let mut w2 = w;
                    let end = if suffix { 7 } else { i + 1 };
                    for x in &mut w2[i..end] {
                        *x += dir * step;
                    }
                    let v = f(w2);
                    if v < best - 1e-5 {
                        best = v;
                        w = w2;
                        improved = true;
                        println!("step {step} -> {best:.5} {w:?}");
                    }
                }
            }
        }
        step /= 2;
    }
    println!(
        "final S={s} L={} R={} lambda={}: {best:.5} {w:?}",
        1 << a.log2_slice_len,
        1 << a.log2_layouts,
        a.bucket_size
    );
}
