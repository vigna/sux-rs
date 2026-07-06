/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Profiling harness for CompVFunc construction.
//!
//! Usage:
//!   sample <pid> 30 -f /tmp/profile.txt

use dsi_progress_logger::no_logging;
use rand::SeedableRng;
use rand::distr::Distribution;
use rand::rngs::SmallRng;
use rand_distr::Geometric;
use sux::func::CompVFunc;
use sux::func::codec::Decoder;

fn main() {
    let n = 10_000_000usize;
    eprintln!("Generating {n} keys + geometric values...");

    let keys: Vec<u64> = (0..n as u64).collect();
    let mut rng = SmallRng::seed_from_u64(0);
    let g = Geometric::new(0.5).unwrap();
    let values: Vec<usize> = (0..n).map(|_| g.sample(&mut rng) as usize + 1).collect();

    eprintln!("Building CompVFunc...");
    let func = CompVFunc::<u64>::try_par_new(&keys, &values, no_logging![]).expect("build");
    eprintln!(
        "Built. max_codeword_len = {}, esym_len  ={}",
        func.decoder().max_codeword_len(),
        func.decoder().escaped_symbols_len()
    );

    // Prevent the optimizer from dropping the build.
    let _ = func.get(0u64);
}
