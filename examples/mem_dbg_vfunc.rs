/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use anyhow::Result;
use clap::Parser;
use epserde::deser::{Deserialize, Flags};
use mem_dbg::{DbgFlags, MemDbg};
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

    <VFunc<usize>>::load_mmap(Args::parse().filename, Flags::default())?
        .mem_dbg(DbgFlags::default() | DbgFlags::FOLLOW_REFS)?;

    Ok(())
}
