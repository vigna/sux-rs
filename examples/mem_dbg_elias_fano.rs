/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use anyhow::Result;
use mem_dbg::{DbgFlags, MemDbg};
use sux::prelude::*;

fn main() -> Result<()> {
    let mut elias_fano_builder = EliasFanoBuilder::new(100_000, 10_000_000);
    for value in 0..100_000 {
        elias_fano_builder.push(value * 100).unwrap();
    }
    // Add an index on ones
    let elias_fano: EliasFano<SelectFixed2> = elias_fano_builder.build().convert_to()?;
    // Add an index on zeros
    let elias_fano: EliasFano<SelectZeroFixed2<SelectFixed2>> = elias_fano.convert_to()?;

    elias_fano.mem_dbg(DbgFlags::default())?;

    let mut v = Vec::<usize>::new();
    v.extend(std::iter::repeat(0).take(100_000));
    let s = v.as_slice();
    s.mem_dbg(DbgFlags::FOLLOW_REFS)?;

    Ok(())
}
