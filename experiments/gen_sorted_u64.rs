/// Generates a binary file of `n` sorted unique u64 values.
/// Usage: gen_sorted_u64 <n> <output_file>
use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} <n> <output_file>", args[0]);
        std::process::exit(1);
    }
    let n: u64 = args[1].parse().expect("invalid n");
    let path = &args[2];

    // Use a simple LCG to generate roughly uniform u64 values, then sort.
    // For very large n we use a spacing approach to avoid allocating n u64s:
    // distribute u64::MAX / n spacing with some jitter.
    eprintln!("Generating {n} sorted unique u64 values...");

    let file = File::create(path).expect("cannot create file");
    let mut w = BufWriter::with_capacity(1 << 20, file);

    // Use deterministic spacing: gap = u64::MAX / n, with small jitter via xorshift
    let gap = u64::MAX / n;
    let mut val: u64 = 0;
    let mut rng_state: u64 = 0xdeadbeefcafebabe;

    for i in 0..n {
        w.write_all(&val.to_ne_bytes()).expect("write failed");

        // Advance with jitter
        let jitter_range = gap / 4;
        // xorshift64
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        let jitter = (rng_state % (jitter_range * 2)).wrapping_sub(jitter_range);
        val = val.saturating_add(gap.wrapping_add(jitter));

        if (i + 1) % 10_000_000 == 0 {
            eprintln!("  wrote {} M keys", (i + 1) / 1_000_000);
        }
    }
    w.flush().expect("flush failed");
    eprintln!("Done. Written {n} keys to {path}");
}
