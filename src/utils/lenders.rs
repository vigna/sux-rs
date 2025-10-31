/*
 *
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 * SPDX-FileCopyrightText: 2025 Inria
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Support for [rewindable fallible lenders](RewindableFallibleLender).
//!
//! Some data structures in this crate have some features in common:
//! - they must be able to read their input more than once;
//! - they might read their input lazily from a source that might generate
//!   errors, such as a file;
//! - they do not store the input they read, but rather some derived data, such
//!   as signatures, so using owned data would be wasteful.
//!
//! For this kind of structures, we provide a [`RewindableFallibleLender`]
//! trait, which is a [`FallibleLender`] with an additional
//! [`rewind`](RewindableFallibleLender::rewind) method that allows rewinding
//! the lender to the beginning.
//!
//! The basic implementation for strings is [`LineLender`], which lends lines
//! from a [`BufRead`] as a `&str`, but lends an internal buffer, rather than
//! allocating a new string for each line. Convenience constructors are provided
//! for [`File`](LineLender::from_file) and [`Path`](LineLender::from_path).
//!  Analogously, we provide `ZstdLineLender` (enabled by the `zstd` feature)
//! that lends lines from a zstd-compressed [`Read`], and [`GzipLineLender`],
//! which lends lines from a gzip-compressed [`Read`].
//!
//! Finally, under the `deko` feature, we provide `DekoLineLender` and
//! `DekoBufLineLender`, which work like the previous two implementations, but
//! detect dynamically the compression format using the
//! [`deko`](https://crates.io/crates/deko) crate.
//! 
//! # Methods
//! 
//! We implement [`RewindableFallibleLender`] on top of most of the adapters
//! returned by methods available for fallible lenders, such as
//! [`map`](lender::FallibleLender::map),
//! [`take`](lender::FallibleLender::take), etc., so when you use these
//! methods you will actually obtain a [`RewindableFallibleLender`].
//! 
//! # Adapters
//!
//! There are several useful adapters available; they often simplify the
//! construction of tests or benchmarks. Almost all adapters implement the
//! [`From`] trait, but at this time due to the complex trait bounds of
//! [`Lender`] type inference rarely works: you'll need to call
//! their `new` methods explicitly.
//!
//! - If you have a slice, [`FromSlice`] is a [`RewindableFallibleLender`] lending its
//!   items. The error of the resulting [`RewindableFallibleLender`] is
//!   [`core::convert::Infallible`].
//!
//! - If you have a cloneable [`IntoIterator`], you can use [`FromCloneableIntoIterator`]
//!   to lend its items; rewinding is implemented by cloning the [`IntoIterator`].
//!   The error of the resulting [`RewindableFallibleLender`] is
//!   [`core::convert::Infallible`].
//!
//! - If you have a type that implements [`IntoLender`] on a reference,
//!   you can use [`FromIntoLender`]; rewinding is implemented
//!   by recreating the lender from the reference. The error
//!   of the resulting [`RewindableFallibleLender`] is
//!   [`core::convert::Infallible`]. If you have instead
//!   a type implementing [`IntoFallibleLender`],
//!   on a reference, use [`FromIntoFallibleLender`] to properly propagate errors.
//!
//! - If you have a type that implements [`IntoIterator`] on a reference,
//!   you can use [`FromIntoIterator`]; rewinding is implemented
//!   by recreating the iterator on the reference. The error
//!   of the resulting [`RewindableFallibleLender`] is
//!   [`core::convert::Infallible`]. If you have instead a type implementing [`IntoFallibleIterator`]
//!   on a reference, use [`FromIntoFallibleIterator`] to properly
//!   propagate errors.
//!
//! - If you have a function that returns an [`IntoLender`] (or an
//!   [`IntoIterator`], via [`lender::IteratorExt::into_lender`]) you can use
//!   [`FromIntoLenderFactory`] to get a [`RewindableFallibleLender`] which will
//!   call that function every time it is rewound. If you have instead a function
//!   returning an [`IntoFallibleLender`], use [`FromIntoFallibleLenderFactory`] to
//!   properly propagate errors.
use fallible_iterator::{FallibleIterator, IntoFallibleIterator};
use flate2::read::GzDecoder;
use io::{BufRead, BufReader};
use lender::{higher_order::FnMutHKARes, *};
use std::{
    fs::File,
    io::{self, Read, Seek},
    path::Path,
};
use zstd::Decoder;

/// The main trait: a [`FallibleLender`] that can be rewound to the beginning.
///
/// Note that [`rewind`](RewindableFallibleLender::rewind) consumes `self` and
/// returns it. This slightly inconvenient behavior is necessary to handle
/// cleanly all implementations, and in particular those involving compression,
/// such as [`ZstdLineLender`] and [`GzipLineLender`].
pub trait RewindableFallibleLender: Sized + FallibleLender {
    /// The type of error happening when rewinding, as distinct
    /// from the error happening when lending.
    type RewindError;
    /// Rewinds the lender to the beginning.
    ///
    /// This method consumes `self` and returns it. This is necessary to handle
    /// cleanly all implementations, and in particular those involving
    /// compression.
    fn rewind(self) -> Result<Self, <Self as RewindableFallibleLender>::RewindError>;
}

// Common next function for all lenders
fn next<'a>(buf: &mut impl BufRead, line: &'a mut String) -> io::Result<Option<&'a str>> {
    line.clear();
    match buf.read_line(line) {
        Err(e) => Err(e),
        Ok(0) => Ok(None),
        Ok(_) => {
            if line.ends_with('\n') {
                line.pop();
                if line.ends_with('\r') {
                    line.pop();
                }
            }
            Ok(Some(line))
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

impl<'lend, B: BufRead> FallibleLending<'lend> for LineLender<B> {
    type Lend = &'lend str;
}

impl<B: BufRead> FallibleLender for LineLender<B> {
    type Error = io::Error;
    fn next(&mut self) -> Result<Option<FallibleLend<'_, Self>>, Self::Error> {
        next(&mut self.buf, &mut self.line)
    }
}

impl<B: BufRead + Seek> RewindableFallibleLender for LineLender<B> {
    type RewindError = io::Error;
    fn rewind(mut self) -> io::Result<Self> {
        self.buf.rewind()?;
        Ok(self)
    }
}

/// A structure lending the lines coming from a
/// [`zstd`](https://facebook.github.io/zstd/)-compressed [`Read`] as `&str`.
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

impl<'lend, R: Read> FallibleLending<'lend> for ZstdLineLender<R> {
    type Lend = &'lend str;
}

impl<R: Read> FallibleLender for ZstdLineLender<R> {
    type Error = io::Error;
    fn next(&mut self) -> Result<Option<FallibleLend<'_, Self>>, Self::Error> {
        next(&mut self.buf, &mut self.line)
    }
}

impl<R: Read + Seek> RewindableFallibleLender for ZstdLineLender<R> {
    type RewindError = io::Error;
    fn rewind(mut self) -> io::Result<Self> {
        let mut read = self.buf.into_inner().finish();
        read.rewind()?;
        self.buf = BufReader::new(Decoder::with_buffer(read)?);
        Ok(self)
    }
}

/// A structure lending the lines coming from a
/// [`gzip`](https://www.gzip.org/)-compressed [`Read`] as `&str`.
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

impl<'lend, R: Read> FallibleLending<'lend> for GzipLineLender<R> {
    type Lend = &'lend str;
}

impl<R: Read> FallibleLender for GzipLineLender<R> {
    type Error = io::Error;
    fn next(&mut self) -> Result<Option<FallibleLend<'_, Self>>, Self::Error> {
        next(&mut self.buf, &mut self.line)
    }
}

impl<R: Read + Seek> RewindableFallibleLender for GzipLineLender<R> {
    type RewindError = io::Error;
    fn rewind(mut self) -> io::Result<Self> {
        let mut read = self.buf.into_inner().into_inner();
        read.rewind()?;
        self.buf = BufReader::new(GzDecoder::new(read));
        Ok(self)
    }
}

#[cfg(feature = "deko")]
mod deko {
    use super::*;
    use std::{
        fs::File,
        io::{self, BufRead, BufReader, Read, Seek},
        path::Path,
    };

    /// A structure lending the lines coming from a compressed [`Read`] as
    /// `&str`.
    ///
    /// The compression format will be detected dynamically by the
    /// [`deko`](https://crates.io/crates/deko) crate.
    ///
    /// The lines are read into a reusable internal string buffer that
    /// grows as needed.
    pub struct DekoLineLender<R: Read> {
        buf: BufReader<::deko::read::AnyDecoder<R>>,
        line: String,
    }

    /// A structure lending the lines coming from a compressed [`BufRead`] as
    /// `&str`.
    ///
    /// The compression format will be detected dynamically by the
    /// [`deko`](https://crates.io/crates/deko) crate.
    ///
    /// The lines are read into a reusable internal string buffer that
    /// grows as needed.
    pub struct DekoBufLineLender<R: BufRead> {
        buf: BufReader<::deko::bufread::AnyDecoder<R>>,
        line: String,
    }

    impl<R: Read> DekoLineLender<R> {
        pub fn new(read: R) -> io::Result<Self> {
            Ok(DekoLineLender {
                buf: BufReader::new(::deko::read::AnyDecoder::new(read)),
                line: String::with_capacity(128),
            })
        }
    }

    impl<R: BufRead> DekoBufLineLender<R> {
        pub fn new(read: R) -> io::Result<Self> {
            Ok(DekoBufLineLender {
                buf: BufReader::new(::deko::bufread::AnyDecoder::new(read)),
                line: String::with_capacity(128),
            })
        }
    }

    impl DekoBufLineLender<BufReader<File>> {
        pub fn from_path(path: impl AsRef<Path>) -> io::Result<DekoBufLineLender<BufReader<File>>> {
            Self::from_file(File::open(path)?)
        }
        pub fn from_file(file: File) -> io::Result<DekoBufLineLender<BufReader<File>>> {
            DekoBufLineLender::new(BufReader::new(file))
        }
    }

    impl<'lend, R: Read> FallibleLending<'lend> for DekoLineLender<R> {
        type Lend = &'lend str;
    }

    impl<'lend, R: BufRead> FallibleLending<'lend> for DekoBufLineLender<R> {
        type Lend = &'lend str;
    }

    impl<R: Read> FallibleLender for DekoLineLender<R> {
        type Error = io::Error;
        fn next(&mut self) -> Result<Option<FallibleLend<'_, Self>>, Self::Error> {
            next(&mut self.buf, &mut self.line)
        }
    }

    impl<R: BufRead> FallibleLender for DekoBufLineLender<R> {
        type Error = io::Error;
        fn next(&mut self) -> Result<Option<FallibleLend<'_, Self>>, Self::Error> {
            next(&mut self.buf, &mut self.line)
        }
    }

    impl<R: Read + Seek> RewindableFallibleLender for DekoLineLender<R> {
        type RewindError = io::Error;
        fn rewind(mut self) -> io::Result<Self> {
            let mut read = self.buf.into_inner().into_inner();
            read.rewind()?;
            self.buf = BufReader::new(::deko::read::AnyDecoder::new(read));
            Ok(self)
        }
    }

    impl<R: BufRead + Seek> RewindableFallibleLender for DekoBufLineLender<R> {
        type RewindError = io::Error;
        fn rewind(mut self) -> io::Result<Self> {
            let mut read = self.buf.into_inner().into_inner();
            read.rewind()?;
            self.buf = BufReader::new(::deko::bufread::AnyDecoder::new(read));
            Ok(self)
        }
    }
}

#[cfg(feature = "deko")]
pub use deko::*;

/// An infallible adapter based on an `AsRef<[T]>`.
///
/// Useful for vectors, slices, etc.
///
/// The functionality of this adapter is subsumed by [`FromIntoIterator`],
/// but this implementation is slightly faster as it does not require
/// storing the returned item internally.
///
/// # Examples
///
/// ```rust
/// # use lender::prelude::*;
/// # use sux::utils::lenders::FromSlice;
/// let data = vec![1, 2, 3];
/// let mut lender = FromSlice::new(data.as_slice());
/// assert_eq!(lender.next()?, Some(&1));
/// assert_eq!(lender.next()?, Some(&2));
/// assert_eq!(lender.next()?, Some(&3));
/// assert_eq!(lender.next()?, None);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct FromSlice<'a, T> {
    slice: &'a [T],
    iter: std::slice::Iter<'a, T>,
}

impl<'a, T> FromSlice<'a, T> {
    pub fn new(slice: &'a [T]) -> Self {
        FromSlice {
            slice,
            iter: slice.iter(),
        }
    }
}

impl<'a, 'lend, T> FallibleLending<'lend> for FromSlice<'a, T> {
    type Lend = &'lend T;
}

impl<'a, T> FallibleLender for FromSlice<'a, T> {
    type Error = core::convert::Infallible;
    fn next(&mut self) -> Result<Option<FallibleLend<'_, Self>>, Self::Error> {
        Ok(self.iter.next())
    }
}

impl<'a, T> RewindableFallibleLender for FromSlice<'a, T> {
    type RewindError = core::convert::Infallible;
    fn rewind(mut self) -> Result<Self, core::convert::Infallible> {
        self.iter = self.slice.as_ref().iter();
        Ok(self)
    }
}

impl<'a, T> From<&'a [T]> for FromSlice<'a, T> {
    fn from(slice: &'a [T]) -> Self {
        FromSlice::new(slice)
    }
}

/// An infallible adapter based on a cloneable [`IntoIterator`].
///
/// Mainly useful for ranges and similar small-footprint types, as rewinding is
/// implemented by cloning the [`IntoIterator`].
///
/// # Examples
///
/// ```
/// # use lender::prelude::*;
/// # use sux::utils::lenders::FromCloneableIntoIterator;
/// let mut lender = FromCloneableIntoIterator::from(0..3);
///
/// assert_eq!(lender.next()?, Some(&0));
/// assert_eq!(lender.next()?, Some(&1));
/// assert_eq!(lender.next()?, Some(&2));
/// assert_eq!(lender.next()?, None);
/// # Ok::<(), Box<dyn std::error::Error>>(())
pub struct FromCloneableIntoIterator<I: IntoIterator> {
    into_iter: I,
    iter: I::IntoIter,
    item: Option<I::Item>,
}

impl<I: IntoIterator + Clone> FromCloneableIntoIterator<I> {
    pub fn new(into_iter: I) -> Self {
        FromCloneableIntoIterator {
            into_iter: into_iter.clone(),
            iter: into_iter.into_iter(),
            item: None,
        }
    }
}

impl<'lend, I: IntoIterator + Clone> FallibleLending<'lend> for FromCloneableIntoIterator<I> {
    type Lend = &'lend I::Item;
}

impl<I: IntoIterator + Clone> FallibleLender for FromCloneableIntoIterator<I> {
    type Error = core::convert::Infallible;
    fn next(&mut self) -> Result<Option<FallibleLend<'_, Self>>, Self::Error> {
        self.item = self.iter.next();
        Ok(self.item.as_ref())
    }
}

impl<I: IntoIterator + Clone> RewindableFallibleLender for FromCloneableIntoIterator<I> {
    type RewindError = core::convert::Infallible;
    fn rewind(mut self) -> Result<Self, Self::RewindError> {
        self.iter = self.into_iter.clone().into_iter();
        Ok(self)
    }
}

impl<I: IntoIterator + Clone> From<I> for FromCloneableIntoIterator<I> {
    fn from(into_iter: I) -> Self {
        FromCloneableIntoIterator::new(into_iter)
    }
}

/// An infallible adapter based on a reference implementing [`IntoLender`].
///
/// Rewinding is implemented by recreating the lender from the reference.
pub struct FromIntoLender<'a, I>
where
    &'a I: IntoLender,
{
    into_lender: &'a I,
    lender: <&'a I as IntoLender>::Lender,
}

impl<'a, I> FromIntoLender<'a, I>
where
    &'a I: IntoLender,
{
    pub fn new(into_lender: &'a I) -> Self {
        FromIntoLender {
            into_lender,
            lender: into_lender.into_lender(),
        }
    }
}

impl<'a, 'lend, I> FallibleLending<'lend> for FromIntoLender<'a, I>
where
    &'a I: IntoLender,
{
    type Lend = Lend<'lend, <&'a I as IntoLender>::Lender>;
}

impl<'a, I> FallibleLender for FromIntoLender<'a, I>
where
    &'a I: IntoLender,
{
    type Error = core::convert::Infallible;
    fn next(&mut self) -> Result<Option<FallibleLend<'_, Self>>, Self::Error> {
        Ok(self.lender.next())
    }
}

impl<'a, I> RewindableFallibleLender for FromIntoLender<'a, I>
where
    &'a I: IntoLender,
{
    type RewindError = core::convert::Infallible;
    fn rewind(mut self) -> Result<Self, Self::RewindError> {
        self.lender = self.into_lender.into_lender();
        Ok(self)
    }
}

impl<'a, I> From<&'a I> for FromIntoLender<'a, I>
where
    &'a I: IntoLender,
{
    fn from(into_lender: &'a I) -> Self {
        FromIntoLender::new(into_lender)
    }
}

/// An adapter based on a reference implementing [`IntoFallibleLender`].
///
/// This adapter is similar to [`FromIntoLender`], but properly propagates
/// errors from the underlying fallible lender instead of wrapping them in an
/// infallible result. Rewinding is implemented by recreating the lender
/// from the reference, so it cannot fail.
pub struct FromIntoFallibleLender<'a, I>
where
    &'a I: IntoFallibleLender,
{
    into_fallible_lender: &'a I,
    lender: <&'a I as IntoFallibleLender>::FallibleLender,
}

impl<'a, I> FromIntoFallibleLender<'a, I>
where
    &'a I: IntoFallibleLender,
{
    pub fn new(into_lender: &'a I) -> Self {
        FromIntoFallibleLender {
            into_fallible_lender: into_lender,
            lender: into_lender.into_fallible_lender(),
        }
    }
}

impl<'a, 'lend, I> FallibleLending<'lend> for FromIntoFallibleLender<'a, I>
where
    &'a I: IntoFallibleLender,
{
    type Lend = <<&'a I as IntoFallibleLender>::FallibleLender as FallibleLending<'lend>>::Lend;
}

impl<'a, I> FallibleLender for FromIntoFallibleLender<'a, I>
where
    &'a I: IntoFallibleLender,
{
    type Error = <&'a I as IntoFallibleLender>::Error;
    fn next(&mut self) -> Result<Option<FallibleLend<'_, Self>>, Self::Error> {
        FallibleLender::next(&mut self.lender)
    }
}

impl<'a, I> RewindableFallibleLender for FromIntoFallibleLender<'a, I>
where
    &'a I: IntoFallibleLender,
{
    type RewindError = core::convert::Infallible;
    fn rewind(mut self) -> Result<Self, Self::RewindError> {
        self.lender = self.into_fallible_lender.into_fallible_lender();
        Ok(self)
    }
}

impl<'a, I> From<&'a I> for FromIntoFallibleLender<'a, I>
where
    &'a I: IntoFallibleLender,
{
    fn from(into_lender: &'a I) -> Self {
        FromIntoFallibleLender::new(into_lender)
    }
}

/// An infallible adapter based on a reference implementing [`IntoIterator`].
///
/// Rewinding is implemented by calling [`into_iter()`](IntoIterator::into_iter)
/// on the reference again.
///
/// If your reference is to a slice, consider using [`FromSlice`], which is
/// slightly more efficient, instead.
pub struct FromIntoIterator<'a, I>
where
    &'a I: IntoIterator,
{
    iterable: &'a I,
    iter: <&'a I as IntoIterator>::IntoIter,
    item: Option<<&'a I as IntoIterator>::Item>,
}

impl<'a, I> FromIntoIterator<'a, I>
where
    &'a I: IntoIterator,
{
    pub fn new(iterable: &'a I) -> Self {
        FromIntoIterator {
            iterable,
            iter: iterable.into_iter(),
            item: None,
        }
    }
}

impl<'a, 'lend, I> FallibleLending<'lend> for FromIntoIterator<'a, I>
where
    &'a I: IntoIterator,
{
    type Lend = &'lend <&'a I as IntoIterator>::Item;
}

impl<'a, I> FallibleLender for FromIntoIterator<'a, I>
where
    &'a I: IntoIterator,
{
    type Error = core::convert::Infallible;
    fn next(&mut self) -> Result<Option<FallibleLend<'_, Self>>, Self::Error> {
        self.item = self.iter.next();
        Ok(self.item.as_ref())
    }
}

impl<'a, I> RewindableFallibleLender for FromIntoIterator<'a, I>
where
    &'a I: IntoIterator,
{
    type RewindError = core::convert::Infallible;
    fn rewind(mut self) -> Result<Self, Self::RewindError> {
        self.iter = self.iterable.into_iter();
        Ok(self)
    }
}

impl<'a, I> From<&'a I> for FromIntoIterator<'a, I>
where
    &'a I: IntoIterator,
{
    fn from(iterable: &'a I) -> Self {
        FromIntoIterator::new(iterable)
    }
}

/// An adapter based on a reference implementing [`IntoFallibleIterator`].
///
/// Rewinding is implemented by calling
/// [`into_fallible_iter()`](IntoFallibleIterator::into_fallible_iter) on the
/// reference again, so rewinding cannot fail.
pub struct FromIntoFallibleIterator<'a, I>
where
    &'a I: IntoFallibleIterator,
{
    iterable: &'a I,
    iter: <&'a I as IntoFallibleIterator>::IntoFallibleIter,
    item: Option<<&'a I as IntoFallibleIterator>::Item>,
}

impl<'a, I> FromIntoFallibleIterator<'a, I>
where
    &'a I: IntoFallibleIterator,
{
    pub fn new(iterable: &'a I) -> Self {
        FromIntoFallibleIterator {
            iterable,
            iter: iterable.into_fallible_iter(),
            item: None,
        }
    }
}

impl<'a, 'lend, I> FallibleLending<'lend> for FromIntoFallibleIterator<'a, I>
where
    &'a I: IntoFallibleIterator,
{
    type Lend = &'lend <&'a I as IntoFallibleIterator>::Item;
}

impl<'a, I> FallibleLender for FromIntoFallibleIterator<'a, I>
where
    &'a I: IntoFallibleIterator,
{
    type Error = <&'a I as IntoFallibleIterator>::Error;
    fn next(&mut self) -> Result<Option<FallibleLend<'_, Self>>, Self::Error> {
        self.iter.next().map(|value| {
            self.item = value;
            self.item.as_ref()
        })
    }
}

impl<'a, I> RewindableFallibleLender for FromIntoFallibleIterator<'a, I>
where
    &'a I: IntoFallibleIterator,
{
    type RewindError = core::convert::Infallible;
    fn rewind(mut self) -> Result<Self, Self::RewindError> {
        self.iter = self.iterable.into_fallible_iter();
        Ok(self)
    }
}

impl<'a, I> From<&'a I> for FromIntoFallibleIterator<'a, I>
where
    &'a I: IntoFallibleIterator,
{
    fn from(iterable: &'a I) -> Self {
        FromIntoFallibleIterator::new(iterable)
    }
}

/// An adapter based on a function returning lenders.
///
/// Rewinding is implemented by calling the function again (and
/// this is the only time this adapter can fail).
///
/// # Examples
///
/// ```
/// # use lender::prelude::*;
/// # use lender::FromIntoIter;
/// # use core::ops::Range;
/// # use sux::utils::lenders::RewindableFallibleLender;
/// # use sux::utils::lenders::FromIntoLenderFactory;
/// # use sux::utils::lenders::FromCloneableIntoIterator;
/// let mut count = 0;
/// let mut lender = FromIntoLenderFactory::new(|| {
///    count += 1;
///   Ok::<FromIntoIter<Range<i32>>, core::convert::Infallible>((0..count).into_into_lender())
/// })?;
///
/// assert_eq!(lender.next()?, Some(0));
/// assert_eq!(lender.next()?, None);
/// lender = lender.rewind()?;
/// assert_eq!(lender.next()?, Some(0));
/// assert_eq!(lender.next()?, Some(1));
/// assert_eq!(lender.next()?, None);
/// # Ok::<(), Box<dyn std::error::Error>>(())
pub struct FromIntoLenderFactory<L: IntoLender, E, F: FnMut() -> Result<L, E>> {
    f: F,
    lender: L::Lender,
}

impl<L: IntoLender, E, F: FnMut() -> Result<L, E>> FromIntoLenderFactory<L, E, F> {
    pub fn new(mut f: F) -> Result<Self, E> {
        f().map(|lender| FromIntoLenderFactory {
            lender: lender.into_lender(),
            f,
        })
    }
}

impl<'lend, L: IntoLender, E, F: FnMut() -> Result<L, E>> FallibleLending<'lend>
    for FromIntoLenderFactory<L, E, F>
{
    type Lend = <L::Lender as Lending<'lend>>::Lend;
}

impl<L: IntoLender, E, F: FnMut() -> Result<L, E>> FallibleLender
    for FromIntoLenderFactory<L, E, F>
{
    type Error = core::convert::Infallible;

    fn next(&mut self) -> Result<Option<Lend<'_, L::Lender>>, Self::Error> {
        Ok(self.lender.next())
    }
}

impl<L: IntoLender, E, F: FnMut() -> Result<L, E>> RewindableFallibleLender
    for FromIntoLenderFactory<L, E, F>
{
    type RewindError = E;

    fn rewind(mut self) -> Result<Self, Self::RewindError> {
        self.lender = (self.f)()?.into_lender();
        Ok(self)
    }
}

/* Errors with:
 *  error[E0119]: conflicting implementations of trait `std::convert::TryFrom<_>` for type `utils::lenders::FromIntoLenderFactory<_, _, _, _>`

 *  = note: conflicting implementation in crate `core`:
 *          - impl<T, U> std::convert::TryFrom<U> for T
 *            where U: std::convert::Into<T>;
impl<
        L: IntoLender,
        E,
        F: FnMut() -> Result<L, E>,
    > TryFrom<F> for FromIntoLenderFactory<L, E, F>
{
    type Error = E;

    fn try_from(f: F) -> Result<Self, Self::Error> {
        Self::new(f)
    }
}
*/

/// An adapter based on a function returning fallible lenders.
///
/// Rewinding is implemented by calling the function again.
///
/// # Examples
///
/// ```
/// # use core::ops::Range;
/// # use lender::prelude::*;
/// # use lender::FromFallibleIter;
/// # use fallible_iterator::{IntoFallible, IteratorExt};
/// # use sux::utils::lenders::RewindableFallibleLender;
/// # use sux::utils::lenders::FromFallibleIntoLenderFactory;
/// # use sux::utils::lenders::FromCloneableIntoIterator;
/// let mut count = 0;
/// let mut lender = FromFallibleIntoLenderFactory::new(|| {
///   count += 1;
///   Ok::<FromFallibleIter<IntoFallible<Range<i32>>>, core::convert::Infallible>((0..count).into_iter().into_fallible().into_fallible_lender())
/// })?;
///
/// assert_eq!(lender.next()?, Some(0));
/// assert_eq!(lender.next()?, None);
/// lender = lender.rewind()?;
/// assert_eq!(lender.next()?, Some(0));
/// assert_eq!(lender.next()?, Some(1));
/// assert_eq!(lender.next()?, None);
/// # Ok::<(), Box<dyn std::error::Error>>(())
pub struct FromIntoFallibleLenderFactory<L: IntoFallibleLender, E, F: FnMut() -> Result<L, E>> {
    f: F,
    lender: L::FallibleLender,
}

impl<L: FallibleLender, E, F: FnMut() -> Result<L, E>> FromIntoFallibleLenderFactory<L, E, F> {
    pub fn new(mut f: F) -> Result<Self, E> {
        f().map(|lender| FromIntoFallibleLenderFactory { lender, f })
    }
}

impl<'lend, L: FallibleLender, E, F: FnMut() -> Result<L, E>> FallibleLending<'lend>
    for FromIntoFallibleLenderFactory<L, E, F>
{
    type Lend = <L as FallibleLending<'lend>>::Lend;
}

impl<L: FallibleLender, E, F: FnMut() -> Result<L, E>> FallibleLender
    for FromIntoFallibleLenderFactory<L, E, F>
{
    type Error = <L as FallibleLender>::Error;

    fn next(&mut self) -> Result<Option<FallibleLend<'_, L>>, Self::Error> {
        self.lender.next()
    }
}

impl<L: FallibleLender, E, F: FnMut() -> Result<L, E>> RewindableFallibleLender
    for FromIntoFallibleLenderFactory<L, E, F>
{
    type RewindError = E;

    fn rewind(mut self) -> Result<Self, E> {
        self.lender = (self.f)()?.into_fallible_lender();
        Ok(self)
    }
}
impl<
    A: FallibleLender + for<'lend> FallibleLending<'lend> + RewindableFallibleLender,
    B: RewindableFallibleLender<RewindError = A::RewindError, Error = A::Error>
        + for<'lend> FallibleLending<'lend, Lend = FallibleLend<'lend, A>>,
> RewindableFallibleLender for Chain<A, B>
{
    type RewindError = A::RewindError;
    fn rewind(self) -> Result<Self, Self::RewindError> {
        let (a, b) = self.into_inner();
        let b = b.rewind()?;
        let a = a.rewind()?;
        Ok(a.chain(b))
    }
}

impl<L: RewindableFallibleLender + Clone> RewindableFallibleLender for lender::Cycle<L> {
    type RewindError = L::RewindError;
    fn rewind(self) -> Result<Self, Self::RewindError> {
        let (original, _current) = self.into_inner();
        original.rewind().map(|lender| lender.cycle())
    }
}

impl<E, L: ?Sized + for<'all> FallibleLending<'all>> RewindableFallibleLender
    for lender::EmptyFallible<E, L>
{
    type RewindError = core::convert::Infallible;
    fn rewind(self) -> Result<Self, Self::RewindError> {
        Ok(self)
    }
}

impl<L: RewindableFallibleLender> RewindableFallibleLender for lender::Enumerate<L> {
    type RewindError = L::RewindError;
    fn rewind(self) -> Result<Self, Self::RewindError> {
        let lender = self.into_inner();
        lender.rewind().map(|lender| lender.enumerate())
    }
}

impl<L: RewindableFallibleLender> RewindableFallibleLender for lender::Fuse<L> {
    type RewindError = L::RewindError;
    fn rewind(self) -> Result<Self, Self::RewindError> {
        let lender = self.into_inner();
        lender.rewind().map(|lender| lender.fuse())
    }
}

impl<'this, L: RewindableFallibleLender> RewindableFallibleLender
    for lender::FallibleIntersperse<'this, L>
where
    for<'lend> <L as FallibleLending<'lend>>::Lend: Clone,
{
    type RewindError = L::RewindError;
    fn rewind(self) -> Result<Self, Self::RewindError> {
        let (lender, separator) = self.into_parts();
        lender.rewind().map(|lender| lender.intersperse(separator))
    }
}

impl<'this, L: RewindableFallibleLender> RewindableFallibleLender
    for lender::FalliblePeekable<'this, L>
{
    type RewindError = L::RewindError;
    fn rewind(self) -> Result<Self, Self::RewindError> {
        let lender = self.into_inner();
        lender.rewind().map(|lender| lender.peekable())
    }
}

impl<'this, L: RewindableFallibleLender> RewindableFallibleLender
    for lender::FallibleRepeat<'this, <L as FallibleLender>::Error, L>
where
    <L as FallibleLender>::Error: Clone,
    for<'lend> <L as FallibleLending<'lend>>::Lend: Clone,
{
    type RewindError = L::RewindError;
    fn rewind(self) -> Result<Self, Self::RewindError> {
        Ok(self)
    }
}

impl<L: RewindableFallibleLender> RewindableFallibleLender for lender::Skip<L> {
    type RewindError = L::RewindError;
    fn rewind(self) -> Result<Self, Self::RewindError> {
        let (lender, n) = self.into_parts();
        lender.rewind().map(|lender| lender.skip(n))
    }
}

impl<L: RewindableFallibleLender> RewindableFallibleLender for lender::StepBy<L> {
    type RewindError = L::RewindError;
    fn rewind(self) -> Result<Self, Self::RewindError> {
        let (lender, step) = self.into_parts();
        lender.rewind().map(|lender| lender.step_by(step))
    }
}

impl<L: RewindableFallibleLender> RewindableFallibleLender for lender::Take<L> {
    type RewindError = L::RewindError;
    fn rewind(self) -> Result<Self, Self::RewindError> {
        let (lender, n) = self.into_parts();
        lender.rewind().map(|lender| lender.take(n))
    }
}

impl<
    L: RewindableFallibleLender,
    F: for<'all> FnMutHKARes<
            'all,
            <L as lender::FallibleLending<'all>>::Lend,
            <L as lender::FallibleLender>::Error,
        >,
> RewindableFallibleLender for lender::Map<L, F>
{
    type RewindError = L::RewindError;
    fn rewind(self) -> Result<Self, Self::RewindError> {
        let (lender, f) = self.into_parts();
        lender.rewind().map(|lender| lender.map(f))
    }
}

impl<
    'this,
    L: RewindableFallibleLender
        + for<'lend> FallibleLending<'lend, Lend: IntoFallibleLender<Error = L::Error>>,
> RewindableFallibleLender for lender::FallibleFlatten<'this, L>
{
    type RewindError = L::RewindError;
    fn rewind(self) -> Result<Self, Self::RewindError> {
        let lender = self.into_inner();
        lender.rewind().map(|lender| lender.flatten())
    }
}
