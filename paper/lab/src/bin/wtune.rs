//! Coordinate-descent tuning of the bucket priority weights of PHast-R.
//!
//! Usage: wtune <seed bits> <log2 slice len> <repair depth> <bucket size> [<keys> [<wrap multiplier>]]
//!
//! Space does not depend on the hardware, so tuning can be run anywhere.
use lab::phast_r_bits_per_key;
use sux::func::PHastRBuilder;

fn eval(keysets: &[Vec<u64>], b: &PHastRBuilder, s: u32) -> f64 {
    let mut tot = 0.0;
    for keys in keysets {
        tot += phast_r_bits_per_key(keys, b, s);
    }
    tot / keysets.len() as f64
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let s: u32 = args[1].parse().unwrap();
    let ll: u32 = args[2].parse().unwrap();
    let depth: u32 = args[3].parse().unwrap();
    let lam: f64 = args[4].parse().unwrap();
    let n: usize = args.get(5).map(|x| x.parse().unwrap()).unwrap_or(4_000_000);
    let wrap: u32 = args.get(6).map(|x| x.parse().unwrap()).unwrap_or(0);
    let keysets: Vec<Vec<u64>> = (0..2u64)
        .map(|k| {
            (0..n as u64)
                .map(|i| (i + k * n as u64).wrapping_mul(0x9e3779b97f4a7c15) ^ 0xabcdef)
                .collect()
        })
        .collect();
    let base = PHastRBuilder::default()
        .seed_bits(s)
        .log2_slice_len(ll)
        .repair_depth(depth)
        .bucket_size(lam)
        .wrap(wrap);
    // Start from the current defaults of sux (those for (8, 512) and
    // (10, 2048) are the result of a previous run of this program)
    let mut w: [i64; 7] = match (wrap, s, 1usize << ll) {
        // With wrapping, the weights of PHast+ with wrapping (multiplier 3)
        (1.., 8, 1024) => [-50649, 63792, 110014, 139267, 161285, 176594, 188305],
        (1.., 8, 512) => [25100, 89361, 117113, 134755, 147369, 154606, 172378],
        (1.., 8, 2048) => [-3427, 10388, 90470, 141895, 179413, 208576, 232553],
        (1.., 10, 2048) => [-3301, 12449, 83323, 139924, 169323, 198105, 212187],
        (_, s, l) => match (s, l) {
            (8, 512) => [-48137, 68016, 105189, 121129, 132794, 140850, 145685],
            (9, 1024) => [-60439, 49207, 121850, 149181, 166713, 179181, 187815],
            (10, 2048) => [-3419, 3042, 88860, 135429, 176433, 198538, 214441],
            (11, 4096) => [-2674, 19194, 37310, 111428, 167443, 205425, 236469],
            _ => [-50000, 50000, 100000, 130000, 150000, 165000, 175000],
        },
    };
    let mut best = eval(&keysets, &base.clone().weights(w), s);
    println!("start {best:.5} {w:?}");
    let mut step = 40000i64;
    while step >= 2500 {
        let mut improved = true;
        while improved {
            improved = false;
            for i in 0..7 {
                for dir in [-1i64, 1] {
                    let mut w2 = w;
                    w2[i] += dir * step;
                    let v = eval(&keysets, &base.clone().weights(w2), s);
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
        "final S={s} L={} d={depth} lambda={lam} wrap={wrap}: {best:.5} {w:?}",
        1 << ll
    );
}
