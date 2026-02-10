/*
 *
 * SPDX-FileCopyrightText: 2024 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

#![cfg(target_pointer_width = "64")]

use clap::{Parser, ValueEnum};
use mem_dbg::*;
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use sux::{
    bits::BitVec,
    rank_sel::{Rank9, RankSmall, Select9, SelectAdapt},
    traits::*,
};

#[derive(Parser)]
#[command(about = "Prints the memory layout of rank and select structures.", long_about = None)]
struct Args {
    len: usize,
    density: f64,
    #[arg(short, long)]
    non_uniform: bool,
    #[arg(value_enum)]
    sel_type: StructType,
}

trait Struct {
    fn build(bits: BitVec) -> Self;
}
impl Struct for SelectAdapt<AddNumBits<BitVec>> {
    fn build(bits: BitVec) -> Self {
        SelectAdapt::new(bits.into(), 3)
    }
}
impl Struct for Select9 {
    fn build(bits: BitVec) -> Self {
        Select9::new(Rank9::new(bits))
    }
}

impl Struct for Rank9 {
    fn build(bits: BitVec) -> Self {
        Rank9::new(bits)
    }
}

impl Struct for RankSmall<2, 9> {
    fn build(bits: BitVec) -> Self {
        RankSmall::<2, 9>::new(bits)
    }
}

impl Struct for RankSmall<1, 9> {
    fn build(bits: BitVec) -> Self {
        RankSmall::<1, 9>::new(bits)
    }
}

impl Struct for RankSmall<1, 10> {
    fn build(bits: BitVec) -> Self {
        RankSmall::<1, 10>::new(bits)
    }
}

impl Struct for RankSmall<1, 11> {
    fn build(bits: BitVec) -> Self {
        RankSmall::<1, 11>::new(bits)
    }
}

impl Struct for RankSmall<3, 13> {
    fn build(bits: BitVec) -> Self {
        RankSmall::<3, 13>::new(bits)
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

    let first_half = loop {
        let b = (0..len / 2)
            .map(|_| rng.random_bool(density0))
            .collect::<BitVec>();
        if b.count_ones() > 0 {
            break b;
        }
    };
    let second_half = (0..len / 2)
        .map(|_| rng.random_bool(density1))
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
    let args = Args::parse();

    let uniform = !args.non_uniform;

    match args.sel_type {
        StructType::SelectAdapt => {
            mem_usage::<SelectAdapt<_>>(args.len, args.density, uniform, "SelectAdapt");
        }
        StructType::Select9 => {
            mem_usage::<Select9>(args.len, args.density, uniform, "Select9");
        }
        StructType::Rank9 => {
            mem_usage::<Rank9>(args.len, args.density, uniform, "Rank9");
        }
        StructType::RankSmall0 => {
            mem_usage::<RankSmall<2, 9>>(args.len, args.density, uniform, "RankSmall0");
        }
        StructType::RankSmall1 => {
            mem_usage::<RankSmall<1, 9>>(args.len, args.density, uniform, "RankSmall1");
        }
        StructType::RankSmall2 => {
            mem_usage::<RankSmall<1, 10>>(args.len, args.density, uniform, "RankSmall2");
        }
        StructType::RankSmall3 => {
            mem_usage::<RankSmall<1, 11>>(args.len, args.density, uniform, "RankSmall3");
        }
        StructType::RankSmall4 => {
            mem_usage::<RankSmall<3, 13>>(args.len, args.density, uniform, "RankSmall4");
        }
    }
}
