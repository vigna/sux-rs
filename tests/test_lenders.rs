/*
 * SPDX-FileCopyrightText: 2025 Inria
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use std::fmt::Debug;
use std::io::{Cursor, Write};

use anyhow::{Context, Result, bail, ensure};
use fallible_iterator::{FallibleIterator, IntoFallibleIterator};
use flate2::write::GzEncoder;
use lender::{FallibleLender, FallibleLending, IntoFallibleLender};

use sux::utils::lenders::*;

fn test_lender<
    L: RewindableFallibleLender<
            RewindError: Debug + std::error::Error + Send + Sync + 'static,
            Error: Debug + std::error::Error + Send + Sync + 'static,
        > + for<'lend> FallibleLending<'lend, Lend = &'lend (impl ?Sized + AsRef<str>)>,
>(
    mut lender: L,
) -> Result<()> {
    for pass in 0..5 {
        for i in 0..3 {
            match lender.next() {
                Ok(Some(got)) => {
                    let got = got.as_ref();
                    let expected = ["foo", "bar", "baz"][i];
                    ensure!(
                        got == expected,
                        "Mismatch of item {i} of pass {pass}: expected {expected:?}, got {got:?}"
                    );
                }
                Err(e) => bail!("Could not read item {i} of pass {pass}: {e:?}"),
                Ok(None) => bail!("Found only {i} items at pass {pass}"),
            }
        }
        if let Some(_) = lender.next()? {
            bail!("Found extra item after pass {pass}");
        }

        lender = lender.rewind()?;
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
fn test_gzip_line_lender() -> Result<()> {
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
    test_lender(FromCloneableIntoIterator::from(["foo", "bar", "baz"]))?;
    test_lender(FromSlice::new(["foo", "bar", "baz"].as_slice()))?;

    // Test From trait implementation for FromSlice
    test_lender(FromSlice::from(["foo", "bar", "baz"].as_slice()))?;

    // Test FromIterableRef with a Vec (where &Vec implements IntoIterator)
    let vec = vec!["foo", "bar", "baz"];
    test_lender(FromIntoIterator::new(&vec))?;
    // Test From trait for FromIterableRef
    test_lender(FromIntoIterator::from(&vec))?;

    // Test FromIterableRef with an array (where &[T] implements IntoIterator)
    let array = ["foo", "bar", "baz"];
    test_lender(FromIntoIterator::new(&array))?;
    // Test From trait for FromIterableRef
    test_lender(FromIntoIterator::from(&array))?;

    Ok(())
}

// Test support for FromIntoFallibleLender
struct FallibleVecLender<T> {
    vec: Vec<T>,
    index: usize,
}

impl<'lend, T> FallibleLending<'lend> for FallibleVecLender<T> {
    type Lend = &'lend T;
}

impl<T> FallibleLender for FallibleVecLender<T> {
    type Error = std::io::Error;
    fn next(&mut self) -> Result<Option<&T>, std::io::Error> {
        if self.index < self.vec.len() {
            let item = &self.vec[self.index];
            self.index += 1;
            Ok(Some(item))
        } else {
            Ok(None)
        }
    }
}

struct VecWrapper {
    vec: Vec<&'static str>,
}

impl<'a> IntoFallibleLender for &'a VecWrapper {
    type Error = std::io::Error;
    type FallibleLender = FallibleVecLender<&'static str>;
    fn into_fallible_lender(self) -> Self::FallibleLender {
        FallibleVecLender {
            vec: self.vec.clone(),
            index: 0,
        }
    }
}

#[test]
fn test_from_into_fallible_lender() -> Result<()> {
    let wrapper = VecWrapper {
        vec: vec!["foo", "bar", "baz"],
    };
    test_lender(FromIntoFallibleLender::new(&wrapper))?;
    // Test From trait for FromIntoFallibleLender
    test_lender(FromIntoFallibleLender::from(&wrapper))?;
    Ok(())
}

// Test support for FromIntoFallibleIterator
struct FallibleVecIter {
    items: Vec<&'static str>,
    index: usize,
}

impl FallibleIterator for FallibleVecIter {
    type Item = &'static str;
    type Error = std::io::Error;

    fn next(&mut self) -> Result<Option<Self::Item>, Self::Error> {
        if self.index < self.items.len() {
            let item = self.items[self.index];
            self.index += 1;
            Ok(Some(item))
        } else {
            Ok(None)
        }
    }
}

struct FallibleVec {
    items: Vec<&'static str>,
}

impl<'a> IntoFallibleIterator for &'a FallibleVec {
    type Item = &'static str;
    type Error = std::io::Error;
    type IntoFallibleIter = FallibleVecIter;

    fn into_fallible_iter(self) -> Self::IntoFallibleIter {
        FallibleVecIter {
            items: self.items.clone(),
            index: 0,
        }
    }
}

#[test]
fn test_from_into_fallible_iterator() -> Result<()> {
    let fallible_vec = FallibleVec {
        items: vec!["foo", "bar", "baz"],
    };
    test_lender(FromIntoFallibleIterator::new(&fallible_vec))?;
    // Test From trait for FromIntoFallibleIterator
    test_lender(FromIntoFallibleIterator::from(&fallible_vec))?;
    Ok(())
}

/*
#[test]
fn test_from_into_lender_factory() -> Result<()> {
    test_lender(
        FromIntoLenderFactory::new(|| -> Result<_, std::io::Error> {
            Ok(["foo", "bar", "baz"].into_into_lender())
        })
        .context("Could not initialize lender")?,
    )
}
*/

/*
#[test]
fn test_from_result_lender_factory() -> Result<()> {
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
*/
