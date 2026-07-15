/*
 * SPDX-FileCopyrightText: 2024 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use mem_dbg::*;
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use sux::{
    bits::BitVec,
    rank_sel::{Rank9, RankSmall, Select9, SelectAdapt},
    traits::*,
};

#[derive(Parser)]
#[command(about = "Prints the memory layout of rank and select structures.", long_about = None, next_line_help = true, max_term_width = 100)]
struct Args {
    /// The length of the bit vector (at least 2).​
    #[arg(value_parser = parse_len)]
    len: usize,
    /// The finite density of ones in the bit vector, in (0, 1].​
    #[arg(value_parser = parse_density)]
    density: f64,
    /// Scale the density by 1% in the first half and 99% in the second.​
    #[arg(short, long)]
    non_uniform: bool,
    /// The rank/select structure to test.​
    #[arg(value_enum)]
    sel_type: StructType,
}

fn parse_len(value: &str) -> Result<usize> {
    let len = value
        .parse::<usize>()
        .with_context(|| format!("invalid bit-vector length '{value}'"))?;
    if len < 2 {
        bail!("bit-vector length must be at least 2");
    }
    Ok(len)
}

fn parse_density(value: &str) -> Result<f64> {
    let density = value
        .parse::<f64>()
        .with_context(|| format!("invalid density '{value}'"))?;
    if !(density.is_finite() && 0.0 < density && density <= 1.0) {
        bail!("density must be finite and in (0, 1]");
    }
    Ok(density)
}

trait Struct {
    fn build(bits: BitVec<Vec<u64>>) -> Self;
}
impl Struct for SelectAdapt<AddNumBits<BitVec<Vec<u64>>>> {
    fn build(bits: BitVec<Vec<u64>>) -> Self {
        SelectAdapt::new(bits.into())
    }
}
impl Struct for Select9<Rank9<BitVec<Vec<u64>>>> {
    fn build(bits: BitVec<Vec<u64>>) -> Self {
        Select9::new(Rank9::<BitVec<Vec<u64>>>::new(bits))
    }
}

impl Struct for Rank9<BitVec<Vec<u64>>> {
    fn build(bits: BitVec<Vec<u64>>) -> Self {
        Rank9::<BitVec<Vec<u64>>>::new(bits)
    }
}

impl Struct for RankSmall<64, 2, 9, BitVec<Vec<u64>>> {
    fn build(bits: BitVec<Vec<u64>>) -> Self {
        RankSmall::<64, 2, 9, BitVec<Vec<u64>>>::new(bits)
    }
}

impl Struct for RankSmall<64, 1, 9, BitVec<Vec<u64>>> {
    fn build(bits: BitVec<Vec<u64>>) -> Self {
        RankSmall::<64, 1, 9, BitVec<Vec<u64>>>::new(bits)
    }
}

impl Struct for RankSmall<64, 1, 10, BitVec<Vec<u64>>> {
    fn build(bits: BitVec<Vec<u64>>) -> Self {
        RankSmall::<64, 1, 10, BitVec<Vec<u64>>>::new(bits)
    }
}

impl Struct for RankSmall<64, 1, 11, BitVec<Vec<u64>>> {
    fn build(bits: BitVec<Vec<u64>>) -> Self {
        RankSmall::<64, 1, 11, BitVec<Vec<u64>>>::new(bits)
    }
}

impl Struct for RankSmall<64, 3, 13, BitVec<Vec<u64>>> {
    fn build(bits: BitVec<Vec<u64>>) -> Self {
        RankSmall::<64, 3, 13, BitVec<Vec<u64>>>::new(bits)
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
enum StructType {
    Rank9,
    RankSmall0,
    RankSmall1,
    RankSmall2,
    RankSmall3,
    RankSmall4,
    SelectAdapt,
    Select9,
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

    let mut first_half = (0..len / 2)
        .map(|_| rng.random_bool(density0))
        .collect::<BitVec<Vec<u64>>>();
    if first_half.count_ones() == 0 {
        first_half.set(0, true);
    }

    let second_half = (0..len - len / 2)
        .map(|_| rng.random_bool(density1))
        .collect::<BitVec<Vec<u64>>>();

    let bits = first_half
        .into_iter()
        .chain(&second_half)
        .collect::<BitVec<Vec<u64>>>();

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
    use num_primitive::PrimitiveNumber;

    let overhead_bits = (s.mem_size(SizeFlags::default()) * 8).saturating_sub(s.len());
    overhead_bits.as_to::<f64>() * 100.0 / s.len().as_to::<f64>()
}

fn main() {
    let args = Args::parse();

    let uniform = !args.non_uniform;

    match args.sel_type {
        StructType::SelectAdapt => {
            mem_usage::<SelectAdapt<AddNumBits<BitVec<Vec<u64>>>>>(
                args.len,
                args.density,
                uniform,
                "SelectAdapt",
            );
        }
        StructType::Select9 => {
            mem_usage::<Select9<Rank9<BitVec<Vec<u64>>>>>(
                args.len,
                args.density,
                uniform,
                "Select9",
            );
        }
        StructType::Rank9 => {
            mem_usage::<Rank9<BitVec<Vec<u64>>>>(args.len, args.density, uniform, "Rank9");
        }
        StructType::RankSmall0 => {
            mem_usage::<RankSmall<64, 2, 9, BitVec<Vec<u64>>>>(
                args.len,
                args.density,
                uniform,
                "RankSmall0",
            );
        }
        StructType::RankSmall1 => {
            mem_usage::<RankSmall<64, 1, 9, BitVec<Vec<u64>>>>(
                args.len,
                args.density,
                uniform,
                "RankSmall1",
            );
        }
        StructType::RankSmall2 => {
            mem_usage::<RankSmall<64, 1, 10, BitVec<Vec<u64>>>>(
                args.len,
                args.density,
                uniform,
                "RankSmall2",
            );
        }
        StructType::RankSmall3 => {
            mem_usage::<RankSmall<64, 1, 11, BitVec<Vec<u64>>>>(
                args.len,
                args.density,
                uniform,
                "RankSmall3",
            );
        }
        StructType::RankSmall4 => {
            mem_usage::<RankSmall<64, 3, 13, BitVec<Vec<u64>>>>(
                args.len,
                args.density,
                uniform,
                "RankSmall4",
            );
        }
    }
}
