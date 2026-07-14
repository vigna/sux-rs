/*
 * SPDX-FileCopyrightText: 2025 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use std::fs;

use anyhow::{Context, Result, bail};
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
    /// Require the map to be a permutation and invert it.​
    #[arg(short, long)]
    invert: bool,
}

fn build<const SORTED: bool>(args: &Args, file: &str) -> Result<()> {
    // SAFETY: ε-serde validates its storage envelope; representation validity
    // is an explicit precondition of this unsafe deserialization API.
    let rcl = unsafe { <RearCodedListStr<SORTED>>::load_full(&args.rcl)? };
    let len = rcl.len();
    let mut values = Vec::with_capacity(len);
    for (line_index, line) in file.lines().enumerate() {
        let value = line.trim().parse::<usize>().with_context(|| {
            format!(
                "cannot parse mapping value on line {} of '{}'",
                line_index + 1,
                args.map
            )
        })?;
        if value >= len {
            bail!(
                "mapping value {value} on line {} is out of range for {len} strings",
                line_index + 1
            );
        }
        values.push(value);
    }
    if values.len() != len {
        bail!(
            "mapping length mismatch: expected {len} values, found {}",
            values.len()
        );
    }

    if args.invert {
        let mut inverse = vec![usize::MAX; len];
        for (index, value) in values.into_iter().enumerate() {
            if inverse[value] != usize::MAX {
                bail!("mapping is not a permutation: value {value} appears more than once");
            }
            inverse[value] = index;
        }
        values = inverse;
    }

    let width = usize::try_from(len.bit_len()).expect("mapping bit width must fit into usize");
    let mut map = bit_field_vec![width => 0; len];
    for (index, value) in values.into_iter().enumerate() {
        map.set_value(index, value);
    }

    let mapped = MappedRearCodedListStr::<SORTED>::from_parts(rcl, map.into());
    // SAFETY: `mapped` was built through validated safe constructors, so all
    // ε-serde representation invariants hold.
    unsafe { mapped.store(&args.dest)? };
    Ok(())
}

fn main() -> Result<()> {
    init_env_logger()?;

    let args = Args::parse();
    let file = fs::read_to_string(&args.map)
        .with_context(|| format!("cannot read mapping file '{}'", args.map))?;
    if args.unsorted {
        build::<false>(&args, &file)
    } else {
        build::<true>(&args, &file)
    }
}
