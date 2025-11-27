/*
 * SPDX-FileCopyrightText: 2025 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use std::fs;

use anyhow::Result;
use clap::Parser;
use epserde::{deser::Deserialize, ser::Serialize};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use sux::{init_env_logger, prelude::*};
use sync_cell_slice::SyncSlice;

#[derive(Parser, Debug)]
#[command(about = "Serializes a mapped rear-coded list starting from a rear-coded list and a mapping.", long_about = None)]
struct Args {
    /// A file containing an ε-serde serialized rear-coded list.
    rcl: String,
    /// An ASCII file containing the mapping, one value per line.
    map: String,
    /// A name for the ε-serde serialized mapped rear-coded list.
    dest: String,
    /// The rear-coded list is not sorted.
    #[arg(short, long, default_value_t = false)]
    unsorted: bool,
    /// Assume the map is a permutation and invert it.
    #[arg(short, long, default_value_t = false)]
    invert: bool,
}

fn main() -> Result<()> {
    init_env_logger()?;

    let args = Args::parse();

    let file = fs::read_to_string(args.map)?;

    let mut map: Vec<usize> = file
        .lines()
        .filter_map(|line| line.trim().parse::<usize>().ok())
        .collect();

    if args.invert {
        let mut inv_map = vec![0; map.len()];
        let sync_slice = inv_map.as_sync_slice();
        map.par_iter()
            .with_min_len(100000)
            .enumerate()
            .for_each(|(i, &x)| {
                unsafe { sync_slice[x].set(i) };
            });
        map = inv_map;
    }

    let map = map.into_boxed_slice();

    if args.unsorted {
        unsafe {
            let rcl = <RearCodedListStr<false>>::load_full(&args.rcl)?;
            <MappedRearCodedListStr<false>>::from_parts(rcl, map).store(&args.dest)?;
        }
    } else {
        unsafe {
            let rcl = <RearCodedListStr<true>>::load_full(&args.rcl)?;
            <MappedRearCodedListStr<true>>::from_parts(rcl, map).store(&args.dest)?;
        }
    }

    Ok(())
}
