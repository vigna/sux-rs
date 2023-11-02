/*
 *
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

/*!

Utility wrappers for files.

*/

use lender::*;
use std::{io::*, path::Path};
use zstd::stream::read::Decoder;

pub struct LineLender<B> {
    buf: B,
    line: String,
}

impl<'lend, B: BufRead> Lending<'lend> for LineLender<B> {
    type Lend = Result<&'lend String>;
}

impl<B: BufRead> Lender for LineLender<B> {
    fn next<'lend>(&'lend mut self) -> Option<Lend<'_, Self>> {
        self.line.clear();
        match self.buf.read_line(&mut self.line) {
            Err(e) => return Some(Err(e)),
            Ok(0) => return None,
            Ok(_) => (),
        };
        if self.line.ends_with('\n') {
            self.line.pop();
            if self.line.ends_with('\r') {
                self.line.pop();
            }
        }
        Some(Ok(&self.line))
    }
}

/// Adapter to iterate over the lines of a file.
#[derive(Clone)]
pub struct FilenameIntoLender<P: AsRef<Path>>(pub P);

impl<P: AsRef<Path>> IntoLender for FilenameIntoLender<P> {
    type Lender = LineLender<BufReader<std::fs::File>>;

    fn into_lender(self) -> Self::Lender {
        LineLender {
            buf: BufReader::new(std::fs::File::open(self.0).unwrap()),
            line: String::new(),
        }
    }
}

/// Adapter to iterate over the lines of a file compressed with Zstandard.
#[derive(Clone)]
pub struct FilenameZstdIntoLender<P: AsRef<Path>>(pub P);

impl<P: AsRef<Path>> IntoLender for FilenameZstdIntoLender<P> {
    type Lender = LineLender<BufReader<Decoder<'static, BufReader<std::fs::File>>>>;

    fn into_lender(self) -> Self::Lender {
        LineLender {
            buf: BufReader::new(Decoder::new(std::fs::File::open(self.0).unwrap()).unwrap()),
            line: String::new(),
        }
    }
}

#[derive(Clone, Debug)]
#[repr(transparent)]
pub struct IntoOkIterator<I, E>(I, std::marker::PhantomData<E>);

impl<I: IntoIterator, E> IntoIterator for IntoOkIterator<I, E> {
    type Item = Result<I::Item>;
    type IntoIter = std::iter::Map<I::IntoIter, fn(I::Item) -> Result<I::Item>>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter().map(|x| Ok(x))
    }
}

impl<I: IntoIterator, E> From<I> for IntoOkIterator<I, E> {
    fn from(into_iter: I) -> Self {
        IntoOkIterator(into_iter, std::marker::PhantomData)
    }
}

#[derive(Clone, Debug)]
#[repr(transparent)]
pub struct IntoOkLender<I, E>(I, std::marker::PhantomData<E>);

impl<I: IntoIterator, E> IntoLender for IntoOkLender<I, E> {
    type Lender =
        FromIter<std::iter::Map<I::IntoIter, fn(I::Item) -> std::result::Result<I::Item, E>>>;

    fn into_lender(self) -> Self::Lender {
        from_iter(self.0.into_iter().map(|x| Ok(x)))
    }
}

impl<I: IntoIterator, E> From<I> for IntoOkLender<I, E> {
    fn from(into_iter: I) -> Self {
        IntoOkLender(into_iter, std::marker::PhantomData)
    }
}
