#![cfg(target_pointer_width = "64")]
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::env;
use std::hint::black_box;
use std::io::Write;
use sux::{
    bits::BitVec,
    rank_sel::{Rank9, Select9, SimpleSelect},
    traits::{Rank, Select},
};

const LENS: [usize; 6] = [
    1_000_000,
    4_000_000,
    16_000_000,
    64_000_000,
    256_000_000,
    1_024_000_000,
];

const DENSITIES: [f64; 3] = [0.1, 0.5, 0.9];

const REPEATS: usize = 7;

const NUMPOS: usize = 70_000_000;

trait SelStruct<B>: Select {
    fn new(bits: B) -> Self;
}
impl SelStruct<BitVec> for SimpleSelect {
    fn new(bits: BitVec) -> Self {
        SimpleSelect::new(bits, 3)
    }
}
impl SelStruct<BitVec> for Select9 {
    fn new(bits: BitVec) -> Self {
        Select9::new(bits)
    }
}

fn remap128(x: usize, n: usize) -> usize {
    ((x as u128).wrapping_mul(n as u128) >> 64) as usize
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
    let num_ones_first_half = first_half.count_ones();
    let second_half = (0..numbits / 2)
        .map(|_| rng.gen_bool(density1))
        .collect::<BitVec>();
    let num_ones_second_half = second_half.count_ones();
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
                    (num_ones_first_half + remap128(rng.gen::<usize>(), num_ones_second_half))
                        as usize,
                ) as u64
            }
        } else {
            unsafe {
                sel.select_unchecked((remap128(rng.gen::<usize>(), num_ones_first_half)) as usize)
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
            let mut time = 0.0;
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

fn bench_rank9() {
    let mut rng = SmallRng::seed_from_u64(0);
    let mut file = std::fs::File::create("target/bench_like_cpp/rank9.csv").unwrap();

    for len in LENS.iter().copied() {
        for density in DENSITIES.iter().copied() {
            let mut time = 0.0;
            for _ in 0..REPEATS {
                let bits = (0..len).map(|_| rng.gen_bool(density)).collect::<BitVec>();
                let rank9: Rank9<BitVec, Vec<usize>> = Rank9::new(bits);
                let mut u = 0;
                let begin = std::time::Instant::now();
                for _ in 0..NUMPOS {
                    u ^= rank9.rank(remap128(rng.gen::<usize>() ^ u, len) as usize);
                }
                black_box(u);
                let elapsed = begin.elapsed().as_nanos();
                time += elapsed as f64 / NUMPOS as f64;
            }
            time /= REPEATS as f64;
            writeln!(file, "{}, {}, {}", len, density, time).unwrap();
        }
    }
}

fn main() {
    if let Some(core_ids) = core_affinity::get_core_ids() {
        // Not core 0. Anything goes.
        let core_id = core_ids[1];
        if !core_affinity::set_for_current(core_id) {
            eprintln!("Cannot pin thread to core {:?}", core_id);
        }
    } else {
        eprintln!("Cannot retrieve core ids");
    }

    std::fs::create_dir_all("target/bench_like_cpp").unwrap();

    let mut rng = rand::rngs::SmallRng::seed_from_u64(0);

    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("Please provide a benchmark option: 'select' or 'rank'");
        return;
    }

    match args[1].as_str() {
        "select" => {
            bench_select_batch::<SimpleSelect>(&mut rng, "simple_select", true);
            bench_select_batch::<SimpleSelect>(&mut rng, "simple_select_non_uniform", false);
            bench_select_batch::<Select9>(&mut rng, "rank9sel", true);
            bench_select_batch::<Select9>(&mut rng, "rank9sel_non_uniform", false);
        }
        "rank" => {
            bench_rank9();
        }
        _ => {
            println!("Invalid benchmark option: '{}'", args[1]);
        }
    }
}
