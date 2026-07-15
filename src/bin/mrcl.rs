/*
 * SPDX-FileCopyrightText: 2025 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use std::fs::File;
use std::io::{BufRead, BufReader};

use anyhow::{Context, Result, bail};
use clap::Parser;
use epserde::{deser::Deserialize, ser::Serialize};
use sux::traits::{BitVecOps, BitVecOpsMut};
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

fn build<const SORTED: bool>(args: &Args) -> Result<()> {
    // SAFETY: ε-serde validates its storage envelope; representation validity
    // is an explicit precondition of this unsafe deserialization API.
    let rcl = unsafe { <RearCodedListStr<SORTED>>::load_full(&args.rcl)? };
    let len = rcl.len();
    let width = usize::try_from(len.bit_len()).expect("mapping bit width must fit into usize");
    // Write parsed values straight into the packed map while streaming the
    // mapping file, so peak memory is O(packed map) instead of also holding the
    // whole text and an intermediate Vec<usize>. In invert mode a single-bit
    // `seen` vector (len bits) rejects non-permutations without a second
    // len-sized value buffer.
    let mut map = bit_field_vec![width => 0; len];
    let mut seen = BitVec::<Vec<usize>>::new(if args.invert { len } else { 0 });
    let map_file = File::open(&args.map)
        .with_context(|| format!("cannot open mapping file '{}'", args.map))?;
    let mut count = 0usize;
    for (line_index, line) in BufReader::new(map_file).lines().enumerate() {
        let line =
            line.with_context(|| format!("cannot read line {} of '{}'", line_index + 1, args.map))?;
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
        // Fail fast (and stop reading) once the map has more entries than strings.
        if count == len {
            bail!("mapping length mismatch: expected {len} values, found more");
        }
        if args.invert {
            // Invert a permutation: map[value] = index. Reject a repeated target.
            if seen.get(value) {
                bail!("mapping is not a permutation: value {value} appears more than once");
            }
            seen.set(value, true);
            map.set_value(value, line_index);
        } else {
            map.set_value(line_index, value);
        }
        count += 1;
    }
    if count != len {
        bail!("mapping length mismatch: expected {len} values, found {count}");
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
    if args.unsorted {
        build::<false>(&args)
    } else {
        build::<true>(&args)
    }
}
