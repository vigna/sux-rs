/*
 *
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 * SPDX-FileCopyrightText: 2025 Inria
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
//! Finally, under the `deko` feature, we provide `DekoLineLender` and
//! `DekoBufLineLender`, which work like the previous two implementations, but
//! detect dynamically the compression format using the
//! [`deko`](https://crates.io/crates/deko) crate.
//!
//! There are a few useful adapters available; they often simplify the
//! construction of tests or benchmarks:
//!
//! - If you have a slice, [`FromSlice`] is a [`RewindableIoLender`] lending its
//!   items. The error of the resulting [`RewindableIoLender`] is
//!   [`core::convert::Infallible`](core::convert::Infallible).
//!
//! - If you have a clonable [`IntoIterator`], you can use [`FromIntoIterator`]
//!   to lend its items; rewinding is implemented by cloning the [`IntoIterator`].
//!   Note that [`FromIntoIterator`] implements the [`From`] trait, but at this
//!   time due to the complex trait bounds of [`Lender`] type inference rarely
//!   works; you'll need to call [`FromIntoIterator::from`] explicitly. The error
//!   of the resulting [`RewindableIoLender`] is
//!   [`core::convert::Infallible`](core::convert::Infallible).
//!
//! - If you have a value that implements [`IntoLender`] on a reference,
//!   you can use [`FromLender`] to lend its items; rewinding is implemented
//!   by recreating the lender from the reference. The same considerations of
//!   [`FromLender`] apply.
//!
//! - If you have a function that returns a [`Lender`] (or an [`IntoIterator`], via
//!   [`lender::IteratorExt::into_lender`]) you can use [`FromLenderFactory`] or
//!   [`FromResultLenderFactory`] to make get a [`RewindableIoLender`], which will
//!   call that function every time it is rewound. Again, the consideration above apply.
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
    type Error: Into<anyhow::Error> + Send + Sync + 'static;
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
        read.rewind()?;
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

    impl<'lend, R: Read> Lending<'lend> for DekoLineLender<R> {
        type Lend = io::Result<&'lend str>;
    }

    impl<'lend, R: BufRead> Lending<'lend> for DekoBufLineLender<R> {
        type Lend = io::Result<&'lend str>;
    }

    impl<R: Read> Lender for DekoLineLender<R> {
        fn next(&mut self) -> Option<<Self as Lending<'_>>::Lend> {
            next(&mut self.buf, &mut self.line)
        }
    }

    impl<R: BufRead> Lender for DekoBufLineLender<R> {
        fn next(&mut self) -> Option<<Self as Lending<'_>>::Lend> {
            next(&mut self.buf, &mut self.line)
        }
    }

    impl<R: Read + Seek> RewindableIoLender<str> for DekoLineLender<R> {
        type Error = io::Error;
        fn rewind(mut self) -> io::Result<Self> {
            let mut read = self.buf.into_inner().into_inner();
            read.rewind()?;
            self.buf = BufReader::new(::deko::read::AnyDecoder::new(read));
            Ok(self)
        }
    }

    impl<R: BufRead + Seek> RewindableIoLender<str> for DekoBufLineLender<R> {
        type Error = io::Error;
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

/// An infallible adapter lending the items of an `AsRef<[T]>`.
///
/// Useful for vectors, slices, etc.
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

impl<'a, 'lend, T> Lending<'lend> for FromSlice<'a, T> {
    type Lend = Result<&'lend T, core::convert::Infallible>;
}

impl<'a, 'lend, T> Lender for FromSlice<'a, T> {
    fn next(&mut self) -> Option<Lend<'_, Self>> {
        self.iter.next().map(Ok)
    }
}

impl<'a, T> RewindableIoLender<T> for FromSlice<'a, T> {
    type Error = core::convert::Infallible;
    fn rewind(mut self) -> Result<Self, Self::Error> {
        self.iter = self.slice.as_ref().iter();
        Ok(self)
    }
}

/// An adapter lending the items of a clonable [`IntoIterator`].
///
/// Mainly useful for ranges and similar small-footprint types, as rewinding is
/// implemented by cloning the iterator.
pub struct FromIntoIterator<I: IntoIterator> {
    into_iter: I,
    iter: I::IntoIter,
    item: Option<I::Item>,
}

impl<I: IntoIterator + Clone> FromIntoIterator<I> {
    pub fn new(into_iter: I) -> Self {
        FromIntoIterator {
            into_iter: into_iter.clone(),
            iter: into_iter.into_iter(),
            item: None,
        }
    }
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
        FromIntoIterator::new(into_iter)
    }
}

/// An adapter lending the items of lenders returned by a reference
/// implementing [`IntoLender`].
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

impl<'a, 'lend, I> Lending<'lend> for FromIntoLender<'a, I>
where
    &'a I: IntoLender,
{
    type Lend = Result<Lend<'lend, <&'a I as IntoLender>::Lender>, core::convert::Infallible>;
}

impl<'a, I> Lender for FromIntoLender<'a, I>
where
    &'a I: IntoLender,
{
    fn next(&mut self) -> Option<Lend<'_, Self>> {
        self.lender.next().map(Ok)
    }
}

impl<'a, T, I> RewindableIoLender<T> for FromIntoLender<'a, I>
where
    &'a I: IntoLender,
    for<'all> <&'a I as IntoLender>::Lender: Lending<'all, Lend = &'all T>,
{
    type Error = core::convert::Infallible;
    fn rewind(mut self) -> Result<Self, Self::Error> {
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

/// An adapter lending the items of a function returning lenders.
pub struct FromLenderFactory<
    T: Send + Sync,
    L: Lender,
    E: Into<anyhow::Error> + Send + Sync + 'static,
    F: FnMut() -> Result<L, E>,
> {
    f: F,
    lender: L,
    item: Option<T>,
}

impl<
    T: Send + Sync,
    L: Lender,
    E: Into<anyhow::Error> + Send + Sync + 'static,
    F: FnMut() -> Result<L, E>,
> FromLenderFactory<T, L, E, F>
{
    pub fn new(mut f: F) -> Result<Self, E> {
        f().map(|lender| FromLenderFactory {
            lender,
            f,
            item: None,
        })
    }
}

impl<
    'lend,
    T: Send + Sync,
    L: Lender,
    E: Into<anyhow::Error> + Send + Sync + 'static,
    F: FnMut() -> Result<L, E>,
> Lending<'lend> for FromLenderFactory<T, L, E, F>
{
    type Lend = Result<&'lend T, E>;
}

impl<
    T: Send + Sync,
    L: Lender<Lend = T>,
    E: Into<anyhow::Error> + Send + Sync + 'static,
    F: FnMut() -> Result<L, E>,
> Lender for FromLenderFactory<T, L, E, F>
{
    fn next(&mut self) -> Option<Lend<'_, Self>> {
        self.item = self.lender.next();
        self.item.as_ref().map(Ok)
    }
}

impl<
    T: Send + Sync,
    L: Lender<Lend = T>,
    E: Into<anyhow::Error> + Send + Sync + 'static,
    F: FnMut() -> Result<L, E>,
> RewindableIoLender<T> for FromLenderFactory<T, L, E, F>
{
    type Error = E;
    fn rewind(mut self) -> Result<Self, Self::Error> {
        self.lender = (self.f)()?;
        Ok(self)
    }
}

/// An adapter lending the items of a function returning lenders of results.
pub struct FromResultLenderFactory<
    T: Send + Sync,
    L: Lender,
    E: Into<anyhow::Error> + Send + Sync + 'static,
    F: FnMut() -> Result<L, E>,
> {
    f: F,
    lender: L,
    item: Option<Result<T, E>>,
}

impl<
    T: Send + Sync,
    L: Lender,
    E: Into<anyhow::Error> + Send + Sync + 'static,
    F: FnMut() -> Result<L, E>,
> FromResultLenderFactory<T, L, E, F>
{
    pub fn new(mut f: F) -> Result<Self, E> {
        f().map(|lender| FromResultLenderFactory {
            lender,
            f,
            item: None,
        })
    }
}

impl<
    'lend,
    T: Send + Sync,
    L: Lender,
    E: Into<anyhow::Error> + Send + Sync + 'static,
    F: FnMut() -> Result<L, E>,
> Lending<'lend> for FromResultLenderFactory<T, L, E, F>
{
    type Lend = Result<&'lend T, E>;
}

impl<
    T: Send + Sync,
    L: Lender,
    E: Into<anyhow::Error> + Send + Sync + 'static,
    F: FnMut() -> Result<L, E>,
> Lender for FromResultLenderFactory<T, L, E, F>
where
    for<'lend> L: Lending<'lend, Lend = Result<T, E>>,
{
    fn next(&mut self) -> Option<Lend<'_, Self>> {
        self.item = self.lender.next();
        match self.item {
            Some(Ok(ref item)) => Some(Ok(item)),
            Some(Err(_)) => Some(
                self.item
                    .take()
                    .unwrap()
                    .map(|_| unreachable!("self.item was Err, but now it's Ok")),
            ),
            None => None,
        }
    }
}

impl<
    T: Send + Sync,
    L: Lender,
    E: Into<anyhow::Error> + Send + Sync + 'static,
    F: FnMut() -> Result<L, E>,
> RewindableIoLender<T> for FromResultLenderFactory<T, L, E, F>
where
    for<'lend> L: Lending<'lend, Lend = Result<T, E>>,
{
    type Error = E;
    fn rewind(mut self) -> Result<Self, Self::Error> {
        self.lender = (self.f)()?;
        Ok(self)
    }
}

/* Errors with:
 *  error[E0119]: conflicting implementations of trait `std::convert::TryFrom<_>` for type `utils::lenders::FromLenderFactory<_, _, _, _>`

 *  = note: conflicting implementation in crate `core`:
 *          - impl<T, U> std::convert::TryFrom<U> for T
 *            where U: std::convert::Into<T>;
impl<
        T: Send + Sync,
        L: Lender,
        E: Into<anyhow::Error> + Send + Sync + 'static,
        F: FnMut() -> Result<L, E>,
    > TryFrom<F> for FromLenderFactory<T, L, E, F>
{
    type Error = E;

    fn try_from(f: F) -> Result<Self, Self::Error> {
        Self::new(f)
    }
}
*/

impl<
    T: ?Sized,
    E: std::error::Error + Send + Sync + 'static,
    A: RewindableIoLender<T, Error = E>,
    B: RewindableIoLender<T, Error = E>,
> RewindableIoLender<T> for lender::Chain<A, B>
{
    type Error = E;

    fn rewind(self) -> Result<Self, E> {
        let (a, b) = self.into_inner();
        let b = b.rewind()?;
        let a = a.rewind()?;
        Ok(a.chain(b))
    }
}

/* can't be implemented because Cloned<L> is an iterator, not a lender
impl<T: Clone, L: RewindableIoLender<T>> RewindableIoLender<T> for lender::Cloned<L> {
    type Error = L::Error;

    fn rewind(self) -> Result<Self, Self::Error> {
        let lender = self.into_inner();
        lender.rewind().map(|lender| lender.cloned())
    }
}

impl<T: Copy, L: RewindableIoLender<T>> RewindableIoLender<T> for lender::Copied<L> {
    type Error = L::Error;

    fn rewind(self) -> Result<Self, Self::Error> {
        let lender = self.into_inner();
        lender.rewind().map(|lender| lender.copied())
    }
}
*/

impl<T: ?Sized, L: RewindableIoLender<T> + Clone> RewindableIoLender<T> for lender::Cycle<L> {
    type Error = L::Error;

    fn rewind(self) -> Result<Self, Self::Error> {
        let (original, _current) = self.into_inner();
        original.rewind().map(|lender| lender.cycle())
    }
}

/* doesn't type-check
impl<T: ?Sized, E> RewindableIoLender<T> for lender::Empty<Result<T, E>> where for<'all> Result<T, E>: lender::Lending<'all>{
    type Error = E;

    fn rewind(self) -> Result<Self, Self::Error> {
        Ok(self)
    }
}
*/

/* would need to yield (usize, Result<T>) instead of Result<(usize, T)>
impl<T: ?Sized, L: RewindableIoLender<T>> RewindableIoLender<(usize, T)> for lender::Enumerate<L> {
    type Error = L::Error;

    fn rewind(self) -> Result<Self, Self::Error> {
        let lender = self.into_inner();
        lender.rewind().map(|lender| lender.enumerate())
    }
}
*/

impl<T: ?Sized, L: RewindableIoLender<T>> RewindableIoLender<T> for lender::Fuse<L> {
    type Error = L::Error;

    fn rewind(self) -> Result<Self, Self::Error> {
        let lender = self.into_inner();
        lender.rewind().map(|lender| lender.fuse())
    }
}

impl<'this, T: ?Sized + 'this, L: RewindableIoLender<T, Error: Clone>> RewindableIoLender<T>
    for lender::Intersperse<'this, L>
{
    type Error = L::Error;

    fn rewind(self) -> Result<Self, Self::Error> {
        let (lender, separator) = self.into_parts();
        lender.rewind().map(|lender| lender.intersperse(separator))
    }
}

impl<'this, T: ?Sized, L: RewindableIoLender<T>> RewindableIoLender<T>
    for lender::Peekable<'this, L>
{
    type Error = L::Error;

    fn rewind(self) -> Result<Self, Self::Error> {
        let lender = self.into_inner();
        lender.rewind().map(|lender| lender.peekable())
    }
}

impl<'this, T: ?Sized, L: RewindableIoLender<T, Error: Clone>> RewindableIoLender<T>
    for lender::Repeat<'this, L>
{
    type Error = L::Error;

    fn rewind(self) -> Result<Self, Self::Error> {
        Ok(self)
    }
}

impl<T: ?Sized, L: RewindableIoLender<T>> RewindableIoLender<T> for lender::Skip<L> {
    type Error = L::Error;

    fn rewind(self) -> Result<Self, Self::Error> {
        let (lender, n) = self.into_parts();
        lender.rewind().map(|lender| lender.skip(n))
    }
}

impl<T: ?Sized, L: RewindableIoLender<T>> RewindableIoLender<T> for lender::StepBy<L> {
    type Error = L::Error;

    fn rewind(self) -> Result<Self, Self::Error> {
        let (lender, step) = self.into_parts();
        lender.rewind().map(|lender| lender.step_by(step))
    }
}

impl<T: ?Sized, L: RewindableIoLender<T>> RewindableIoLender<T> for lender::Take<L> {
    type Error = L::Error;

    fn rewind(self) -> Result<Self, Self::Error> {
        let (lender, n) = self.into_parts();
        lender.rewind().map(|lender| lender.take(n))
    }
}
