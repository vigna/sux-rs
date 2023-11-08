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
    fs::File,
    io::{self, Read, Seek, SeekFrom},
    mem::MaybeUninit,
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

impl LineLender<BufReader<File>> {
    pub fn from_path(path: impl AsRef<Path>) -> io::Result<LineLender<BufReader<File>>> {
        Ok(LineLender::new(BufReader::new(File::open(path)?)))
    }

    pub fn from_file(file: File) -> LineLender<BufReader<File>> {
        LineLender::new(BufReader::new(file))
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

impl<B: BufRead + Seek> RewindableIOLender<str> for LineLender<B> {
    fn rewind(&mut self) -> io::Result<()> {
        self.buf.seek(io::SeekFrom::Start(0)).map(|_| ())
    }
}

pub struct ZstdLineLender<R: Read> {
    buf: BufReader<Decoder<'static, BufReader<R>>>,
    line: String,
}

impl<R: Read> ZstdLineLender<R> {
    pub fn new(read: R) -> io::Result<Self> {
        Ok(ZstdLineLender {
            buf: BufReader::new(Decoder::new(read)?),
            line: String::with_capacity(128),
        })
    }
}

impl ZstdLineLender<BufReader<Decoder<'static, BufReader<File>>>> {
    pub fn from_path(path: impl AsRef<Path>) -> io::Result<ZstdLineLender<File>> {
        ZstdLineLender::new(File::open(path)?)
    }

    pub fn from_file(file: File) -> io::Result<ZstdLineLender<File>> {
        ZstdLineLender::new(file)
    }
}

impl<'lend, R: Read> Lending<'lend> for ZstdLineLender<R> {
    type Lend = io::Result<&'lend str>;
}

impl<R: Read> Lender for ZstdLineLender<R> {
    fn next(&mut self) -> Option<<Self as Lending<'_>>::Lend> {
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

impl<R: Read + Seek> RewindableIOLender<str> for ZstdLineLender<R> {
    fn rewind(&mut self) -> io::Result<()> {
        #[allow(invalid_value)]
        let buf = std::mem::replace(&mut self.buf, unsafe {
            MaybeUninit::uninit().assume_init()
        });
        let mut read = buf.into_inner().finish();
        read.seek(SeekFrom::Current(0)).map(|_| ())?;
        self.buf = BufReader::new(Decoder::with_buffer(read)?);
        Ok(())
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
