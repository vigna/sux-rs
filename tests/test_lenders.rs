/*
 * SPDX-FileCopyrightText: 2025 Inria
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use std::fmt::Debug;
use std::io::{Cursor, Write};

use anyhow::{Context, Result, bail, ensure};
use flate2::write::GzEncoder;
use lender::{IteratorExt, Lender};

use sux::utils::lenders::*;

fn test_lender<T: ?Sized + AsRef<str>, L: RewindableIoLender<T, Error: Debug>>(
    mut lender: L,
) -> Result<()> {
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

        lender = lender
            .rewind()
            .map_err(Into::into)
            .context("Could not rewind")?;
    }

    Ok(())
}

#[test]
fn test_line_lender() -> Result<()> {
    let mut buf = Cursor::new(b"foo\nbar\nbaz\n");
    test_lender(LineLender::new(&mut buf))?;

    #[cfg(feature = "deko")]
    {
        use std::io::BufReader;

        buf.set_position(0);
        test_lender(DekoLineLender::new(&mut buf)?)?;
        buf.set_position(0);
        test_lender(DekoBufLineLender::new(BufReader::new(&mut buf))?)?;
    }
    Ok(())
}

#[test]
fn test_zstd_line_lender() -> Result<()> {
    let mut buf = Cursor::new(
        zstd::stream::encode_all(Cursor::new(b"foo\nbar\nbaz\n"), 1).context("Could not encode")?,
    );
    test_lender(ZstdLineLender::new(&mut buf).context("Could not initialize lender")?)?;

    #[cfg(feature = "deko")]
    {
        use std::io::BufReader;

        buf.set_position(0);
        test_lender(DekoLineLender::new(&mut buf)?)?;
        buf.set_position(0);
        test_lender(DekoBufLineLender::new(BufReader::new(&mut buf))?)?;
    }
    Ok(())
}

#[test]
fn test_gziplinelender() -> Result<()> {
    let mut encoder = GzEncoder::new(Vec::new(), flate2::Compression::default());
    encoder.write_all(b"foo\nbar\nbaz\n")?;
    let mut buf = Cursor::new(encoder.finish().context("Could not encode")?);

    test_lender(GzipLineLender::new(&mut buf).context("Could not initialize lender")?)?;

    #[cfg(feature = "deko")]
    {
        use std::io::BufReader;

        buf.set_position(0);
        test_lender(DekoLineLender::new(&mut buf)?)?;
        buf.set_position(0);
        test_lender(DekoBufLineLender::new(BufReader::new(&mut buf))?)?;
    }
    Ok(())
}

#[test]
fn test_from() -> Result<()> {
    test_lender(FromIntoIterator::from(["foo", "bar", "baz"]))?;
    test_lender(FromSlice::new(["foo", "bar", "baz"].as_slice()))
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

#[test]
fn test_fromresultlenderfactory() -> Result<()> {
    let items = || {
        [
            Ok("foo"),
            Ok("bar"),
            Err(std::io::Error::new(std::io::ErrorKind::NotFound, "error!")),
            Ok("baz"),
        ]
    };
    let mut lender = FromResultLenderFactory::new(|| -> Result<_, std::io::Error> {
        Ok(items().into_iter().into_lender())
    })
    .context("Could not initialize lender")?;

    let items = items();

    for pass in 0..5 {
        for i in 0..4 {
            match lender.next() {
                Some(got) => {
                    let got = got;
                    let expected = &items[i];
                    ensure!(
                        got.is_ok() == expected.is_ok(),
                        "Mismatch of item {i} of pass {pass}: expected {expected:?}, got {got:?}"
                    );
                    if got.is_ok() {
                        ensure!(
                            *got.as_ref().unwrap() == expected.as_ref().unwrap(),
                            "Mismatch of item {i} of pass {pass}: expected {expected:?}, got {got:?}"
                        );
                    }
                }
                None => bail!("Found only {i} items at pass {pass}"),
            }
        }
        if let Some(extra) = lender.next().map(Result::unwrap) {
            bail!("Found extra item after pass {pass}: {}", extra);
        }

        lender = lender.rewind().context("Could not rewind")?;
    }

    Ok(())
}
