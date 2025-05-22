/*
 * SPDX-FileCopyrightText: 2025 Inria
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use std::io::{Cursor, Write};

use anyhow::{bail, ensure, Context, Result};
use flate2::write::GzEncoder;

use sux::utils::lenders::*;

fn test_lender<L: RewindableIoLender<str>>(mut lender: L) -> Result<()> {
    for pass in 0..5 {
        for i in 0..3 {
            match lender.next() {
                Some(Ok(got)) => {
                    let expected = ["foo", "bar", "baz"][i];
                    ensure!(
                        got == expected,
                        "Mismatch of item {i} of pass {pass}: expected {expected:?}, got {got:?}"
                    );
                }
                Some(Err(e)) => bail!("Could not read item {i} of pass {pass}: {e:?}"),
                None => bail!("Found only {i} items at pass {pass}"),
            }
        }
        assert_eq!(lender.next().map(Result::unwrap), None);
        lender = lender.rewind().context("Could not rewind")?;
    }

    Ok(())
}

#[test]
fn test_linelender() -> Result<()> {
    let buf = Cursor::new(b"foo\nbar\nbaz\n");
    test_lender(LineLender::new(buf))
}

#[test]
fn test_zstdlinelender() -> Result<()> {
    let buf = Cursor::new(
        zstd::stream::encode_all(Cursor::new(b"foo\nbar\nbaz\n"), 1).context("Could not encode")?,
    );
    test_lender(ZstdLineLender::new(buf).context("Could not initialize lender")?)
}

#[test]
fn test_gziplinelender() -> Result<()> {
    let mut encoder = GzEncoder::new(Vec::new(), flate2::Compression::default());
    encoder.write_all(b"foo\nbar\nbaz\n")?;
    let buf = Cursor::new(encoder.finish().context("Could not encode")?);

    test_lender(GzipLineLender::new(buf).context("Could not initialize lender")?)
}
