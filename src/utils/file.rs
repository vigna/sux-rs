/*
 *
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

/*!

Utility wrappers for files.

*/

use flate2::read::GzDecoder;
use io::{BufRead, BufReader};
use lender::*;
use std::{
    io::{self, Seek},
    path::Path,
};
use zstd::stream::read::Decoder;

pub trait RewindableIOLender<T: ?Sized>:
    Lender + for<'lend> Lending<'lend, Lend = io::Result<&'lend T>>
{
    fn rewind(&mut self) -> io::Result<()>;
}

/**

A structure lending the lines coming from a [`BufRead`] as `&str`.

The lines are read into a reusable internal string buffer that
grows as needed.

For convenience, we implement [`From`] from [`BufRead`].

*/
pub struct LineLender<B> {
    buf: B,
    line: String,
}

impl<B> LineLender<B> {
    pub fn new(buf: B) -> Self {
        LineLender {
            buf,
            line: String::with_capacity(128),
        }
    }
}

impl<B> From<B> for LineLender<B> {
    fn from(buf: B) -> Self {
        LineLender::new(buf)
    }
}

impl<'lend, B: BufRead> Lending<'lend> for LineLender<B> {
    type Lend = io::Result<&'lend str>;
}

impl<B: BufRead> Lender for LineLender<B> {
    fn next(&mut self) -> Option<Lend<'_, Self>> {
        self.line.clear();
        match self.buf.read_line(&mut self.line) {
            Err(e) => Some(Err(e)),
            Ok(0) => None,
            Ok(_) => {
                if self.line.ends_with('\n') {
                    self.line.pop();
                    if self.line.ends_with('\r') {
                        self.line.pop();
                    }
                }
                Some(Ok(&self.line))
            }
        }
    }
}

impl<B: Seek> Seek for LineLender<B> {
    fn seek(&mut self, pos: io::SeekFrom) -> io::Result<u64> {
        self.buf.seek(pos)
    }
}

impl<B: BufRead + Seek> RewindableIOLender<str> for LineLender<B> {
    fn rewind(&mut self) -> io::Result<()> {
        self.buf.seek(io::SeekFrom::Start(0)).map(|_| ())
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

impl<P: AsRef<Path>> From<P> for FilenameZstdIntoLender<P> {
    fn from(path: P) -> Self {
        FilenameZstdIntoLender(path)
    }
}

/// Adapter to iterate over the lines of a file compressed with Gzip.
#[derive(Clone)]
pub struct FilenameGzipIntoLender<P: AsRef<Path>>(pub P);

impl<P: AsRef<Path>> IntoLender for FilenameGzipIntoLender<P> {
    type Lender = LineLender<BufReader<GzDecoder<std::fs::File>>>;

    fn into_lender(self) -> Self::Lender {
        LineLender {
            buf: BufReader::new(GzDecoder::new(std::fs::File::open(self.0).unwrap())),
            line: String::new(),
        }
    }
}

pub struct FromIntoIterator<I: IntoIterator + Clone> {
    into_iter: I,
    iter: I::IntoIter,
    item: Option<I::Item>,
}

impl<'lend, T: 'lend, I: IntoIterator<Item = T> + Clone> Lending<'lend> for FromIntoIterator<I> {
    type Lend = io::Result<&'lend T>;
}

impl<T: 'static, I: IntoIterator<Item = T> + Clone> Lender for FromIntoIterator<I> {
    fn next(&mut self) -> Option<Lend<'_, Self>> {
        self.item = self.iter.next();
        self.item.as_ref().map(Ok)
    }
}

impl<T: 'static, I: IntoIterator<Item = T> + Clone> RewindableIOLender<T> for FromIntoIterator<I> {
    fn rewind(&mut self) -> io::Result<()> {
        self.iter = self.into_iter.clone().into_iter();
        Ok(())
    }
}

impl<T: 'static, I: IntoIterator<Item = T> + Clone> From<I> for FromIntoIterator<I> {
    fn from(into_iter: I) -> Self {
        FromIntoIterator {
            into_iter: into_iter.clone(),
            iter: into_iter.into_iter(),
            item: None,
        }
    }
}
