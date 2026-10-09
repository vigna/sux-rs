use clap::Parser;
use ph::phast::{
    Conf, DefaultCompressedArray, Function2, GenericCore, ProdOfValues, SeedChooserConf, SeedOnly,
    ShiftOnly, ShiftOnlyProdWrapped, ShiftOnlyWrapped, SumOfValues,
};
use ph::seeds::{Bits8, BitsFast, SeedSize};
use ph::{BuildDefaultSeededHasher, GetSize};
use std::time::Instant;

#[derive(Parser)]
struct Args {
    #[arg(short, default_value_t = 10_000_000)]
    n: usize,
    #[arg(short, default_value_t = 8)]
    s: u8,
    #[arg(short, long, value_delimiter = ',', default_value = "5.25")]
    lambda: Vec<f64>,
    /// phast, plus, w1, w2, w3
    #[arg(short, long, value_delimiter = ',', default_value = "plus")]
    variant: Vec<String>,
    #[arg(short, long, default_value_t = 10_000_000)]
    queries: usize,
}

fn run<SC: SeedChooserConf, SS: SeedSize>(
    keys: &[u64],
    ss: SS,
    b100: u32,
    sc: SC,
    queries: usize,
) -> (f64, f64, f64) {
    let t = Instant::now();
    let f: Function2<GenericCore, SS, SC::Core, DefaultCompressedArray, BuildDefaultSeededHasher> =
        Function2::with_slice_conf_sc(
            keys,
            Conf::generic_with_hash(ss, b100, BuildDefaultSeededHasher::default()),
            sc,
        );
    let build = t.elapsed().as_secs_f64() * 1e9 / keys.len() as f64;
    let bits = f.size_bytes() as f64 * 8.0 / keys.len() as f64;
    // query throughput over random existing keys
    let n = keys.len() as u64;
    let mut x = 0x9e3779b97f4a7c15u64;
    let t = Instant::now();
    let mut acc = 0usize;
    for _ in 0..queries {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let k = keys[((x >> 32) * n >> 32) as usize];
        acc ^= f.get(&k);
    }
    std::hint::black_box(acc);
    let q = t.elapsed().as_secs_f64() * 1e9 / queries as f64;
    (bits, build, q)
}

fn main() {
    let a = Args::parse();
    let keys: Vec<u64> = (0..a.n as u64)
        .map(|i| i.wrapping_mul(0x9e3779b97f4a7c15) ^ 0x1234567)
        .collect();
    for v in &a.variant {
        for &lambda in &a.lambda {
            let b100 = (lambda * 100.0).round() as u32;
            macro_rules! go {
                ($sc:expr) => {
                    if a.s == 8 {
                        run(&keys, Bits8, b100, $sc, a.queries)
                    } else {
                        run(&keys, BitsFast(a.s), b100, $sc, a.queries)
                    }
                };
            }
            let (bits, build, q) = match v.as_str() {
                "phast" => go!(SeedOnly(ProdOfValues)),
                "phastsum" => go!(SeedOnly(SumOfValues)),
                "plus" => go!(ShiftOnly),
                "w1" => go!(ShiftOnlyWrapped::<1>),
                "w2" => go!(ShiftOnlyWrapped::<2>),
                "w3" => go!(ShiftOnlyWrapped::<3>),
                "w1p" => go!(ShiftOnlyProdWrapped::<1>),
                "w2p" => go!(ShiftOnlyProdWrapped::<2>),
                "w3p" => go!(ShiftOnlyProdWrapped::<3>),
                _ => panic!(),
            };
            println!(
                "ref {v} S={} lambda={lambda:.2}: {bits:.4} bits/key, build {build:.1} ns/key, query {q:.1} ns",
                a.s
            );
        }
    }
}
