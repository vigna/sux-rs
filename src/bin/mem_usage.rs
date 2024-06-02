#![cfg(target_pointer_width = "64")]

use clap::{arg, Parser, ValueEnum};
use mem_dbg::*;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use sux::{
    bits::BitVec,
    rank_sel::{Rank10Sel, Rank9, RankSmall, Select9, SimpleSelect},
    traits::*,
};

trait Struct {
    fn build(bits: BitVec) -> Self;
}
impl Struct for SimpleSelect {
    fn build(bits: BitVec) -> Self {
        SimpleSelect::new(bits, 3)
    }
}
impl Struct for Select9 {
    fn build(bits: BitVec) -> Self {
        Select9::new(Rank9::new(bits))
    }
}
impl<const LOG2_UPPER_BLOCK_SIZE: usize, const LOG2_ONES_PER_INVENTORY: usize> Struct
    for Rank10Sel<LOG2_UPPER_BLOCK_SIZE, LOG2_ONES_PER_INVENTORY>
{
    fn build(bits: BitVec) -> Self {
        Rank10Sel::<LOG2_UPPER_BLOCK_SIZE, LOG2_ONES_PER_INVENTORY>::new(bits)
    }
}

impl Struct for Rank9 {
    fn build(bits: BitVec) -> Self {
        Rank9::new(bits)
    }
}

impl<const NUM_U32S: usize> Struct for RankSmall<NUM_U32S> {
    fn build(bits: BitVec) -> Self {
        RankSmall::<NUM_U32S>::new(bits)
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
enum StructType {
    Rank9,
    RankSmall1,
    RankSmall2,
    RankSmall3,
    Simpleselect,
    Select9,
}

#[derive(Parser)]
struct Cli {
    len: usize,
    density: f64,
    #[arg(short, long)]
    non_uniform: bool,
    #[arg(value_enum)]
    sel_type: StructType,
}

fn mem_usage<S: Struct + MemSize + MemDbg + BitLength>(
    len: usize,
    density: f64,
    uniform: bool,
    name: &str,
) {
    let mut rng = SmallRng::seed_from_u64(0);
    let (density0, density1) = if uniform {
        (density, density)
    } else {
        (density * 0.01, density * 0.99)
    };

    let first_half = loop {
        let b = (0..len / 2)
            .map(|_| rng.gen_bool(density0))
            .collect::<BitVec>();
        if b.count_ones() > 0 {
            break b;
        }
    };
    let second_half = (0..len / 2)
        .map(|_| rng.gen_bool(density1))
        .collect::<BitVec>();

    let bits = first_half
        .into_iter()
        .chain(&second_half)
        .collect::<BitVec>();

    let s = S::build(bits);

    let mem_cost = mem_cost(&s);
    println!(
        "BitVec with length: {}, density: {}, uniform: {}",
        len, density, uniform
    );
    println!("Memory cost of {}: {}%", name, mem_cost);
    s.mem_dbg(DbgFlags::PERCENTAGE).unwrap();
}

fn mem_cost<S: Struct + MemSize + MemDbg + BitLength>(s: &S) -> f64 {
    (((s.mem_size(SizeFlags::default()) * 8 - s.len()) * 100) as f64) / (s.len() as f64)
}

fn main() {
    let cli = Cli::parse();

    let uniform = !cli.non_uniform;

    match cli.sel_type {
        StructType::Simpleselect => {
            mem_usage::<SimpleSelect>(cli.len, cli.density, uniform, "SimpleSelect");
        }
        StructType::Select9 => {
            mem_usage::<Select9>(cli.len, cli.density, uniform, "Select9");
        }
        StructType::Rank9 => {
            mem_usage::<Rank9>(cli.len, cli.density, uniform, "Rank9");
        }
        StructType::RankSmall1 => {
            mem_usage::<RankSmall<1>>(cli.len, cli.density, uniform, "RankSmall<1>");
        }
        StructType::RankSmall2 => {
            mem_usage::<RankSmall<2>>(cli.len, cli.density, uniform, "RankSmall<2>");
        }
        StructType::RankSmall3 => {
            mem_usage::<RankSmall<3>>(cli.len, cli.density, uniform, "RankSmall<3>");
        }
    }
}
