/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use anyhow::Result;
use clap::Parser;
use epserde::{
    deser::{Deserialize, Flags},
    ser::Serialize,
};
use mem_dbg::MemDbg;
use sux::prelude::*;

#[derive(Parser, Debug)]
#[command(about = "Prints layout information for an Elias-Fano structure", long_about = None)]
struct Args {
    /// The number of elements
    filename: String,
}

fn main() -> Result<()> {
    stderrlog::new()
        .verbosity(2)
        .timestamp(stderrlog::Timestamp::Second)
        .init()
        .unwrap();

    let e = <EliasFano<SelectZeroFixed2<SelectFixed2>>>::load_mmap(
        Args::parse().filename,
        Flags::default(),
    )?
    .as_ref();

    println!("{}", e.mem_dbg(),);

    Ok(())
}
