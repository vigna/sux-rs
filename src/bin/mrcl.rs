/*
 * SPDX-FileCopyrightText: 2025 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use std::fs;

use anyhow::Result;
use clap::Parser;
use epserde::{deser::Deserialize, ser::Serialize};
use sux::{init_env_logger, prelude::*, utils::PrimitiveUnsignedExt};
use value_traits::slices::SliceByValueMut;

#[derive(Parser, Debug)]
#[command(about = "Serializes a mapped rear-coded list starting from a rear-coded list and a mapping.", long_about = None, next_line_help = true, max_term_width = 100)]
struct Args {
    /// A file containing an ε-serde serialized rear-coded list.​
    rcl: String,
    /// An ASCII file containing the mapping, one value per line.​
    map: String,
    /// A name for the ε-serde serialized mapped rear-coded list.​
    dest: String,
    /// Assume the rear-coded list is not sorted.​
    #[arg(short, long)]
    unsorted: bool,
    /// Assume the map is a permutation and invert it.​
    #[arg(short, long)]
    invert: bool,
}

fn main() -> Result<()> {
    init_env_logger()?;

    let args = Args::parse();

    let file = fs::read_to_string(args.map)?;

    if args.unsorted {
        unsafe {
            let rcl = <RearCodedListStr<false>>::load_full(&args.rcl)?;
            let width = rcl.len().bit_len() as usize;

            let mut map = bit_field_vec![width => 0; rcl.len()];
            for (i, line) in file.lines().enumerate() {
                map.set_value(i, line.trim().parse::<usize>()?);
            }

            if args.invert {
                let mut inv_map = bit_field_vec![width => 0; rcl.len()];
                for (i, v) in map.iter().enumerate() {
                    inv_map.set_value(v, i);
                }

                map = inv_map;
            }

            <MappedRearCodedListStr<false>>::from_parts(rcl, map.into()).store(&args.dest)?;
        }
    } else {
        unsafe {
            let rcl = <RearCodedListStr<true>>::load_full(&args.rcl)?;
            let width = rcl.len().bit_len() as usize;

            let mut map = bit_field_vec![width => 0; rcl.len()];
            for (i, line) in file.lines().enumerate() {
                map.set_value(i, line.trim().parse::<usize>()?);
            }

            if args.invert {
                let mut inv_map = bit_field_vec![width => 0; rcl.len()];
                for (i, v) in map.iter().enumerate() {
                    inv_map.set_value(v, i);
                }

                map = inv_map;
            }

            <MappedRearCodedListStr<true>>::from_parts(rcl, map.into()).store(&args.dest)?;
        }
    }

    Ok(())
}
