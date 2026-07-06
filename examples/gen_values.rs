/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Generate a values file (one ASCII decimal integer per line) of a
//! given length with a specified distribution. Intended as an input
//! for the `comp_vfunc` CLI construction benchmarks.
//!
//! ```text
//! cargo run --release --example gen_values -- \
//!     --n 100000000 --dist geometric --output values_100M.txt
//! ```

use clap::{Parser, ValueEnum};
use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand_distr::{Distribution, Geometric, Zipf};
use std::fs::File;
use std::io::{BufWriter, Write};

#[derive(Copy, Clone, Debug, ValueEnum)]
enum Dist {
    /// Geometric with p=0.5 (entropy ≈ 2 bits).
    Geometric,
    /// Zipf with exponent 1.5 over a million-symbol alphabet.
    Zipf,
    /// Constant 0 (degenerate; all keys map to the same value).
    Zero,
}

#[derive(Parser, Debug)]
#[command(about = "Generate a values file for comp_vfunc benchmarks")]
struct Args {
    /// Number of values to generate.​
    #[arg(short, long)]
    n: usize,
    /// Output file path (ASCII, one integer per line).​
    #[arg(short, long)]
    output: String,
    /// Value distribution.​
    #[arg(long, default_value = "geometric")]
    dist: Dist,
    /// PRNG seed.​
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();
    let mut rng = SmallRng::seed_from_u64(args.seed);
    let file = File::create(&args.output)?;
    let mut w = BufWriter::new(file);
    match args.dist {
        Dist::Geometric => {
            let d = Geometric::new(0.5).unwrap();
            for _ in 0..args.n {
                writeln!(w, "{}", d.sample(&mut rng))?;
            }
        }
        Dist::Zipf => {
            let d = Zipf::<f64>::new(1_000_000.0, 1.5).unwrap();
            for _ in 0..args.n {
                writeln!(w, "{}", (d.sample(&mut rng) as u64).saturating_sub(1))?;
            }
        }
        Dist::Zero => {
            for _ in 0..args.n {
                writeln!(w, "0")?;
            }
        }
    }
    w.flush()?;
    eprintln!("wrote {} values to {}", args.n, args.output);
    Ok(())
}
