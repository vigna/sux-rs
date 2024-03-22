use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::hint::black_box;
use std::io::Write;
use sux::{
    bits::BitVec,
    rank_sel::{Rank9Sel, SimpleSelect},
    traits::Select,
};

const LENS: [u64; 11] = [
    (1u64 << 20) + 2,
    (1 << 21) + 2,
    (1 << 22) + 2,
    (1 << 23) + 2,
    (1 << 24) + 2,
    (1 << 25) + 2,
    (1 << 26) + 2,
    (1 << 27) + 2,
    (1 << 28) + 2,
    (1 << 29) + 2,
    (1 << 30) + 2,
];

const DENSITIES: [f64; 3] = [0.25, 0.5, 0.75];

const REPEATS: usize = 10;

const NUMPOS: usize = 70_000_000;

trait SelStruct<B>: Select {
    fn new(bits: B) -> Self;
}
impl SelStruct<BitVec> for SimpleSelect {
    fn new(bits: BitVec) -> Self {
        SimpleSelect::new(bits, 3)
    }
}
impl SelStruct<BitVec> for Rank9Sel {
    fn new(bits: BitVec) -> Self {
        Rank9Sel::new(bits)
    }
}

fn remap128(x: u64, n: u64) -> u64 {
    ((x as u128).wrapping_mul(n as u128) >> 64) as u64
}

fn bench_select<S: SelStruct<BitVec>>(
    numbits: usize,
    numpos: usize,
    density0: f64,
    density1: f64,
    rng: &mut SmallRng,
) -> f64 {
    let first_half = loop {
        let b = (0..numbits / 2)
            .map(|_| rng.gen_bool(density0))
            .collect::<BitVec>();
        if b.count_ones() > 0 {
            break b;
        }
    };
    let num_ones_first_half = first_half.count_ones() as u64;
    let second_half = (0..numbits / 2)
        .map(|_| rng.gen_bool(density1))
        .collect::<BitVec>();
    let num_ones_second_half = second_half.count_ones() as u64;
    let bits = first_half
        .into_iter()
        .chain(second_half.into_iter())
        .collect::<BitVec>();

    let sel: S = S::new(bits);
    let mut u: u64 = 0;

    let begin = std::time::Instant::now();
    for _ in 0..numpos {
        u ^= if u & 1 != 0 {
            unsafe {
                sel.select_unchecked(
                    (num_ones_first_half + remap128(rng.gen::<u64>(), num_ones_second_half))
                        as usize,
                ) as u64
            }
        } else {
            unsafe {
                sel.select_unchecked((remap128(rng.gen::<u64>(), num_ones_first_half)) as usize)
                    as u64
            }
        };
    }
    let elapsed = begin.elapsed().as_nanos();
    black_box(u);

    elapsed as f64 / numpos as f64
}

fn bench_select_batch<S: SelStruct<BitVec>>(rng: &mut SmallRng, sel_name: &str, uniform: bool) {
    print!("{}... ", sel_name);
    std::io::stdout().flush().unwrap();
    let mut file =
        std::fs::File::create(format!("target/bench_like_cpp/{}.csv", sel_name)).unwrap();
    for (i, len) in LENS.iter().enumerate() {
        for (j, density) in DENSITIES.iter().enumerate() {
            print!(
                "{}/{}\r{}... ",
                i * DENSITIES.len() + j + 1,
                LENS.len() * DENSITIES.len(),
                sel_name
            );
            std::io::stdout().flush().unwrap();
            let (density0, density1) = if uniform {
                (*density, *density)
            } else {
                (*density * 0.01, *density * 0.99)
            };
            let mut time = 0f64;
            for _ in 0..REPEATS {
                time += bench_select::<S>(*len as usize, NUMPOS, density0, density1, rng);
            }
            time /= REPEATS as f64;
            writeln!(file, "{}, {}, {}", len, density, time).unwrap();
        }
    }
    file.flush().unwrap();
    println!("\r{}... done        ", sel_name);
}

fn main() {
    std::fs::create_dir_all("target/bench_like_cpp").unwrap();
    let mut rng = SmallRng::seed_from_u64(0);

    bench_select_batch::<SimpleSelect>(&mut rng, "simple_select", true);
    bench_select_batch::<SimpleSelect>(&mut rng, "simple_select_non_uniform", false);
    bench_select_batch::<Rank9Sel>(&mut rng, "rank9sel", true);
    bench_select_batch::<Rank9Sel>(&mut rng, "rank9sel_non_uniform", false);
}
