/*
 * SPDX-FileCopyrightText: 2025 Inria
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use std::io::{Cursor, Write};

use anyhow::{Context, Result, bail, ensure};
use flate2::write::GzEncoder;
use lender::IteratorExt;

use sux::utils::lenders::*;

fn test_lender<T: ?Sized + AsRef<str>, L: RewindableIoLender<T>>(mut lender: L) -> Result<()> {
    for pass in 0..5 {
        for i in 0..3 {
            match lender.next() {
                Some(Ok(got)) => {
                    let got = got.as_ref();
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
        if let Some(extra) = lender.next().map(Result::unwrap) {
            bail!("Found extra item after pass {pass}: {}", extra.as_ref());
        }

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

#[test]
fn test_fromintoiterator() -> Result<()> {
    test_lender(FromIntoIterator::from(["foo", "bar", "baz"]))
}

#[test]
fn test_fromlenderfactory() -> Result<()> {
    test_lender(
        FromLenderFactory::new(|| -> Result<_, std::io::Error> {
            Ok(["foo", "bar", "baz"].into_iter().into_lender())
        })
        .context("Could not initialize lender")?,
    )
}
