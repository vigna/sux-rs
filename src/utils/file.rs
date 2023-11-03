/*
 *
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

/*!

Utility wrappers for files.

*/

use io::{BufRead, BufReader};
use lender::{higher_order::FnMutHKA, *};
use std::{io, mem::MaybeUninit, path::Path};
use zstd::stream::read::Decoder;

pub struct LineLender<B> {
    buf: B,
    line: String,
}

impl<'lend, B: BufRead> Lending<'lend> for LineLender<B> {
    type Lend = io::Result<&'lend String>;
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
pub struct IntoOkIterator<I>(pub I);

impl<I: IntoIterator> IntoIterator for IntoOkIterator<I> {
    type Item = io::Result<I::Item>;
    type IntoIter = std::iter::Map<I::IntoIter, fn(I::Item) -> io::Result<I::Item>>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter().map(|x| Ok(x))
    }
}

impl<I: IntoIterator> From<I> for IntoOkIterator<I> {
    fn from(into_iter: I) -> Self {
        IntoOkIterator(into_iter)
    }
}

#[derive(Clone, Debug)]
#[repr(transparent)]
pub struct OkLender<I>(pub I);

impl<'lend, I: Lender> Lending<'lend> for OkLender<I> {
    type Lend = io::Result<Lend<'lend, I>>;
}

impl<I: Lender> Lender for OkLender<I> {
    fn next(&mut self) -> Option<Lend<'_, Self>> {
        self.0.next().map(|x| Ok(x))
    }
}

#[derive(Clone, Debug)]
#[repr(transparent)]
pub struct IntoOkLender<I>(pub I);

impl<I: IntoLender> IntoLender for IntoOkLender<I> {
    type Lender = OkLender<I::Lender>;

    fn into_lender(self) -> Self::Lender {
        OkLender(self.0.into_lender())
    }
}

impl<I: IntoLender> From<I> for IntoOkLender<I> {
    fn from(into_iter: I) -> Self {
        IntoOkLender(into_iter)
    }
}

#[derive(Clone, Debug)]
pub struct RefLender<I: Iterator> {
    iter: I,
    item: Option<I::Item>,
}

impl<'lend, I: Iterator> Lending<'lend> for RefLender<I> {
    type Lend = &'lend I::Item;
}

impl<I: Iterator> Lender for RefLender<I> {
    fn next(&mut self) -> Option<Lend<'_, Self>> {
        self.item = self.iter.next();
        self.item.as_ref()
    }
}

#[derive(Clone, Debug)]
#[repr(transparent)]
pub struct IntoRefLender<I: IntoIterator>(pub I);

impl<I: IntoIterator> IntoLender for IntoRefLender<I> {
    type Lender = RefLender<I::IntoIter>;

    fn into_lender(self) -> <Self as IntoLender>::Lender {
        RefLender {
            iter: self.0.into_iter(),
            item: None,
        }
    }
}
