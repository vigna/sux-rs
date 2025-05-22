/*
 *
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Support for [rewindable I/O lenders](RewindableIoLender).
//!
//! Some data structures in this crate have some features in common:
//! - they must be able to read their input more than once;
//! - they might read their input lazily from a source that might generate
//!   errors, such as a file;
//! - they do not store the input they read, but rather some derived data, such
//!   as signatures, so using owned data would be wasteful.
//!
//! For this kind of structures, we provide a [`RewindableIoLender`] trait,
//! which is a [`Lender`] that can be rewound to the beginning, and whose
//! returned items are [`Result`]s. Rewindability solves the first problem,
//! [`Result`]s solve the second problem, while lending solves the third
//! problem.
//!
//! The basic implementation for strings is [`LineLender`], which lends lines
//! from a [`BufRead`] as a `&str`, but lends an internal buffer, rather than
//! allocating a new string for each line. Convenience constructors are provided
//! for [`File`](LineLender::from_file) and [`Path`](LineLender::from_path).
//!  Analogously, we provide `ZstdLineLender` (enabled by the `zstd` feature)
//! that lends lines from a zstd-compressed [`Read`], and [`GzipLineLender`],
//! which lends lines from a gzip-compressed [`Read`].
//!
//! If you have a clonable [`IntoIterator`], you can use [`FromIntoIterator`] to
//! lend its items; rewinding is implemented by cloning the iterator. Note that
//! [`FromIntoIterator`] implements the [`From`] trait, but at this time due to
//! the complex trait bounds of [`Lender`] type inference rarely works; you'll
//! need to call [`FromIntoIterator::from`] explicitly. The error of the resulting
//! [`RewindableIoLender`] is `core::convert::Infallible`.
//!
use flate2::read::GzDecoder;
use io::{BufRead, BufReader};
use lender::*;
use std::{
    fs::File,
    io::{self, Read, Seek},
    path::Path,
};
use zstd::Decoder;

/// The main trait: a [`Lender`] that can be rewound to the beginning, and whose
/// returned item are [`Result`]s.
///
/// Additionally, this trait is implemented on [`lender::Take`], so you can call
/// `take` on a rewindable lender and obtain again a rewindable lender.
///
/// [`LineLender`] is an implementation reading lines from a file, but you can
/// turn any clonable [`IntoIterator`] into a rewindable lender with
/// [`FromIntoIterator`].
///
/// Note that [`rewind`](RewindableIoLender::rewind) consumes `self` and returns
/// it. This slightly inconvenient behavior is necessary to handle cleanly all
/// implementations, and in particular those involving compression, such as
/// [`ZstdLineLender`] and [`GzipLineLender`].
pub trait RewindableIoLender<T: ?Sized>:
    Sized + Lender + for<'lend> Lending<'lend, Lend = Result<&'lend T, Self::Error>>
{
    type Error: std::error::Error + Send + Sync + 'static;
    /// Rewind the lender to the beginning.
    ///
    /// This method consumes `self` and returns it. This is necessary to handle
    /// cleanly all implementations, and in particular those involving
    /// compression.
    fn rewind(self) -> Result<Self, Self::Error>;
}

// Common next function for all lenders
fn next<'a>(buf: &mut impl BufRead, line: &'a mut String) -> Option<io::Result<&'a str>> {
    line.clear();
    match buf.read_line(line) {
        Err(e) => Some(Err(e)),
        Ok(0) => None,
        Ok(_) => {
            if line.ends_with('\n') {
                line.pop();
                if line.ends_with('\r') {
                    line.pop();
                }
            }
            Some(Ok(line))
        }
    }
}

/// A structure lending the lines coming from a [`BufRead`] as `&str`.
///
/// The lines are read into a reusable internal string buffer that grows as
/// needed.
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
        next(&mut self.buf, &mut self.line)
    }
}

impl<B: BufRead + Seek> RewindableIoLender<str> for LineLender<B> {
    type Error = io::Error;
    fn rewind(mut self) -> io::Result<Self> {
        self.buf.rewind()?;
        Ok(self)
    }
}

/// A structure lending the lines coming from a zstd-compressed [`Read`] as
/// `&str`.
///
/// The lines are read into a reusable internal string buffer that grows as
/// needed.
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
        next(&mut self.buf, &mut self.line)
    }
}

impl<R: Read + Seek> RewindableIoLender<str> for ZstdLineLender<R> {
    type Error = io::Error;
    fn rewind(mut self) -> io::Result<Self> {
        let mut read = self.buf.into_inner().finish();
        read.stream_position()?;
        self.buf = BufReader::new(Decoder::with_buffer(read)?);
        Ok(self)
    }
}

/// A structure lending the lines coming from a gzip-compressed [`Read`] as `&str`.
///
/// The lines are read into a reusable internal string buffer that
/// grows as needed.
#[derive(Debug)]
pub struct GzipLineLender<R: Read> {
    buf: BufReader<GzDecoder<R>>,
    line: String,
}

impl<R: Read> GzipLineLender<R> {
    pub fn new(read: R) -> io::Result<Self> {
        Ok(GzipLineLender {
            buf: BufReader::new(GzDecoder::new(read)),
            line: String::with_capacity(128),
        })
    }
}

impl GzipLineLender<BufReader<GzDecoder<BufReader<File>>>> {
    pub fn from_path(path: impl AsRef<Path>) -> io::Result<GzipLineLender<File>> {
        GzipLineLender::new(File::open(path)?)
    }

    pub fn from_file(file: File) -> io::Result<GzipLineLender<File>> {
        GzipLineLender::new(file)
    }
}

impl<'lend, R: Read> Lending<'lend> for GzipLineLender<R> {
    type Lend = io::Result<&'lend str>;
}

impl<R: Read> Lender for GzipLineLender<R> {
    fn next(&mut self) -> Option<<Self as Lending<'_>>::Lend> {
        next(&mut self.buf, &mut self.line)
    }
}

impl<R: Read + Seek> RewindableIoLender<str> for GzipLineLender<R> {
    type Error = io::Error;
    fn rewind(mut self) -> io::Result<Self> {
        let mut read = self.buf.into_inner().into_inner();
        read.stream_position()?;
        self.buf = BufReader::new(GzDecoder::new(read));
        Ok(self)
    }
}

/// An adapter lending the items of a clonable [`IntoIterator`].
pub struct FromIntoIterator<I: IntoIterator + Clone> {
    into_iter: I,
    iter: I::IntoIter,
    item: Option<I::Item>,
}

impl<'lend, T: 'lend, I: IntoIterator<Item = T> + Clone> Lending<'lend> for FromIntoIterator<I> {
    type Lend = Result<&'lend T, core::convert::Infallible>;
}

impl<T: 'static, I: IntoIterator<Item = T> + Clone> Lender for FromIntoIterator<I> {
    fn next(&mut self) -> Option<Lend<'_, Self>> {
        self.item = self.iter.next();
        self.item.as_ref().map(Ok)
    }
}

impl<T: 'static, I: IntoIterator<Item = T> + Clone> RewindableIoLender<T> for FromIntoIterator<I> {
    type Error = core::convert::Infallible;
    fn rewind(mut self) -> Result<Self, Self::Error> {
        self.iter = self.into_iter.clone().into_iter();
        Ok(self)
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

impl<T: ?Sized, L: RewindableIoLender<T>> RewindableIoLender<T> for lender::Take<L> {
    type Error = L::Error;

    fn rewind(self) -> Result<Self, Self::Error> {
        let (lender, n) = self.into_parts();
        lender.rewind().map(|lender| lender.take(n))
    }
}
