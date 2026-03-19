use criterion::{BenchmarkId, Criterion, Throughput};
use mem_dbg::{MemDbg, SizeFlags};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};
use std::hint::black_box;
use sux::bits::BitVec;
use sux::traits::bit_field_slice::Word;
use sux::traits::{BitCount, BitLength, Rank, Select};

mod impls;
pub use impls::*;

pub trait Build<B> {
    fn new(bits: B) -> Self;
}

pub fn create_bitvec<W: Word>(
    rng: &mut SmallRng,
    len: u64,
    density: f64,
    uniform: bool,
) -> (u64, u64, BitVec<Vec<W>>) {
    let (density0, density1) = if uniform {
        (density, density)
    } else {
        (density * 0.01, density * 0.99)
    };

    let len1 = len.div_ceil(2);
    let len2 = len / 2;

    let first_half = loop {
        let b = (0..len1)
            .map(|_| rng.random_bool(density0))
            .collect::<BitVec<Vec<W>>>();
        if b.count_ones() > 0 {
            break b;
        }
    };
    let second_half = (0..len2)
        .map(|_| rng.random_bool(density1))
        .collect::<BitVec<Vec<W>>>();
    let num_ones_second_half = second_half.count_ones() as u64;
    let num_ones_first_half = first_half.count_ones() as u64;

    let bits = first_half
        .into_iter()
        .chain(&second_half)
        .collect::<BitVec<Vec<W>>>();

    (num_ones_first_half, num_ones_second_half, bits)
}

/// Returns the memory overhead of a structure as a percentage of the
/// bit-vector length.
pub fn mem_cost(benched_struct: &(impl MemDbg + BitLength)) -> f64 {
    ((benched_struct.mem_size(SizeFlags::default()) * 8 - benched_struct.len()) * 100) as f64
        / benched_struct.len() as f64
}

/// Writes memory-overhead data as JSON into criterion's output directory
/// at `target/criterion/<name>/mem_cost.json`.
pub fn write_mem_costs(name: &str, mem_costs: &[(u64, f64, f64)]) {
    let dir = format!("target/criterion/{}", name);
    std::fs::create_dir_all(&dir).unwrap();
    let entries: Vec<String> = mem_costs
        .iter()
        .map(|(size, dense, mc)| {
            format!(
                "  {{\"size\": {}, \"dense\": {}, \"mem_cost\": {:.6}}}",
                size, dense, mc
            )
        })
        .collect();
    let json = format!("[\n{}\n]\n", entries.join(",\n"));
    std::fs::write(format!("{}/mem_cost.json", dir), json).unwrap();
}

/// Returns a random rank distributed proportionally to the density in
/// each half of the bit vector.
#[inline(always)]
pub fn random_rank(rng: &mut SmallRng, first_half: u64, second_half: u64) -> u64 {
    if rng.random_bool(0.5) {
        ((rng.random::<u64>() as u128).wrapping_mul(first_half as u128) >> 64) as u64
    } else {
        first_half + ((rng.random::<u64>() as u128).wrapping_mul(second_half as u128) >> 64) as u64
    }
}

fn bench_inner<W: Word, S: Build<BitVec<Vec<W>>> + MemDbg + BitLength>(
    c: &mut Criterion,
    name: &str,
    lengths: &[u64],
    densities: &[f64],
    uniform: bool,
    op: impl Fn(&S, usize) -> usize,
) {
    let name = if uniform {
        name.to_string()
    } else {
        format!("{}_nonuniform", name)
    };
    let mut group = c.benchmark_group(&name);
    group.throughput(Throughput::Elements(1));
    let mut rng = SmallRng::seed_from_u64(0);
    let mut mem_costs = Vec::new();
    for &len in lengths {
        for &density in densities {
            let (num_ones_first_half, num_ones_second_half, bits) =
                create_bitvec::<W>(&mut rng, len, density, uniform);
            let s: S = S::new(bits);
            mem_costs.push((len, density, mem_cost(&s)));
            group.bench_function(
                BenchmarkId::from_parameter(format!("{}_{}", len, density)),
                |b| {
                    b.iter(|| {
                        let r = random_rank(&mut rng, num_ones_first_half, num_ones_second_half);
                        black_box(op(&s, r as usize))
                    })
                },
            );
        }
    }
    group.finish();
    write_mem_costs(&name, &mem_costs);
}

pub fn bench_select<W: Word, S: Build<BitVec<Vec<W>>> + Select + MemDbg + BitLength>(
    c: &mut Criterion,
    name: &str,
    lengths: &[u64],
    densities: &[f64],
    uniform: bool,
) {
    bench_inner::<W, S>(c, name, lengths, densities, uniform, |s, r| unsafe {
        s.select_unchecked(r)
    });
}

pub fn bench_rank<W: Word, R: Build<BitVec<Vec<W>>> + Rank + MemDbg + BitLength>(
    c: &mut Criterion,
    name: &str,
    lengths: &[u64],
    densities: &[f64],
    uniform: bool,
) {
    bench_inner::<W, R>(c, name, lengths, densities, uniform, |s, r| unsafe {
        s.rank_unchecked(r)
    });
}
