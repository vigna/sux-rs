/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Tommaso Fontana
 * SPDX-FileCopyrightText: 2025 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! A [`RearCodedList`] indexed through a mapping.
//!
//! The typical use case of this structure is to compress and access efficiently
//! list of strings that are not in sorted order. You build a sorted
//! [`RearCodedList`], and then you [wrap the instance together with a
//! mapping](MappedRearCodedList::from_parts) that maps the original order
//! to the sorted order.
//!
//! There is no constraint, however, on the mapping: it can also collapse indices;
//! and the rear-coded list can also be unsorted.
//!
//! As in the case of [`RearCodedList`], the structure is parameterized by the
//! input/output type and by the storage backend, but two versions are presently
//! implemented, and associated with a type alias:
//!
//! - [`MappedRearCodedListStr`]: stores UTF-8 elements (`AsRef<str>`),
//!   returning [`String`] on access;
//!
//! - [`MappedRearCodedListSliceU8`]: stores byte sequences (`AsRef<[u8]>`),
//!   returning `Vec<u8>` on access.
//!
//! These standard types use a `Box<[usize]>` for storing the mapping.
//! However, it is possible to use any type implementing the
//! [`SliceByValue`] trait from the [`value_traits`] crate.
//!
//! Besides the standard access by means of the [`IndexedSeq`] trait, this
//! structure also implements the `get_in_place` method, which allows to write
//! the element directly into a user-provided buffer (a string or a vector of
//! bytes), avoiding allocations. [`MappedRearCodedListStr`] has an additional
//! [`get_bytes_in_place`](MappedRearCodedListStr::get_bytes_in_place) method that
//! writes the bytes of the string into a user-provided `Vec<u8>`.
//!
//! Mapped rear-coded lists can be iterated upon using either an
//! [`Iterator`](RearCodedList::iter) or a [`Lender`](RearCodedList::lender). In
//! the first case there will be an allocation at each iteration, whereas in
//! second case a single buffer will be reused. You can also [iterate from a
//! given position](RearCodedList::lender_from). The iteration will not be as
//! fast as in the non-mapped case, however, as it is not possible to build the
//! returned strings incrementally.
//!
//! Note that, contrarily to [`RearCodedList`], this structure does not provide
//! implementations for the [`IndexedDict`](crate::traits::IndexedDict) trait,
//! independently of whether the underlying [`RearCodedList`] is sorted or not.
//!
//! Finally, the `mrcl` command-line tool can be use to create
//! a serialized mapped rear-coded list starting from a
//! serialized rear-coded list and a mapping.
//!
//! # Examples
//!
//! Here we build a sorted rear-coded list, and then we wrap it in a
//! [`MappedRearCodedListStr`] together with a permutation:
//!
//! ```
//! # use sux::traits::IndexedSeq;
//! # use sux::dict::RearCodedListBuilder;
//! # use sux::dict::MappedRearCodedListStr;
//!
//! let mut rclb = RearCodedListBuilder::<str, true>::new(4);
//!
//! rclb.push("aa");
//! rclb.push("aab");
//! rclb.push("abc");
//! rclb.push("abdd");
//! rclb.push("abde\0f");
//! rclb.push("abdf");
//!
//! let rcl = rclb.build();
//! let map = vec![5, 4, 2, 0, 1, 3].into_boxed_slice(); // permutation
//! let mrcl = MappedRearCodedListStr::from_parts(rcl, map);
//! assert_eq!(mrcl.get(0), "abdf");
//! assert_eq!(mrcl.get(1), "abde\0f");
//! assert_eq!(mrcl.get(2), "abc");
//! assert_eq!(mrcl.get(3), "aa");
//! assert_eq!(mrcl.get(4), "aab");
//! assert_eq!(mrcl.get(5), "abdd");
//! ```
use crate::bits::BitFieldVec;
use crate::dict::rear_coded_list::RearCodedList;
use crate::traits::{IndexedSeq, IntoIteratorFrom, Types};
use lender::FusedLender;
use lender::{ExactSizeLender, IntoLender, Lender, Lending};
use mem_dbg::*;
use value_traits::slices::SliceByValue;

/// Main structure; please use the type aliases [`MappedRearCodedListStr`] and
/// [`MappedRearCodedListSliceU8`].
#[derive(Debug, Clone, MemDbg, MemSize)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MappedRearCodedList<
    I: ?Sized,
    O,
    D = Box<[u8]>,
    P = Box<[usize]>,
    Q = BitFieldVec,
    const SORTED: bool = true,
> {
    rcl: RearCodedList<I, O, D, P, SORTED>,
    map: Q,
}

pub type MappedRearCodedListSliceU8<const SORTED: bool = true> =
    MappedRearCodedList<[u8], Vec<u8>, Box<[u8]>, Box<[usize]>, BitFieldVec, SORTED>;
/// A rear-coded list of strings.
pub type MappedRearCodedListStr<const SORTED: bool = true> =
    MappedRearCodedList<str, String, Box<[u8]>, Box<[usize]>, BitFieldVec, SORTED>;

impl<
    I: PartialEq<O> + PartialEq + ?Sized,
    O: PartialEq<I> + PartialEq,
    D: AsRef<[u8]>,
    P: AsRef<[usize]>,
    Q: SliceByValue<Value = usize>,
    const SORTED: bool,
> MappedRearCodedList<I, O, D, P, Q, SORTED>
{
    /// Creates a new instance from its parts.
    ///
    /// It is your responsibility to ensure that the mapping is valid,
    /// that is, that its length matches the length of the rear-coded list,
    /// and that all indices are in range.
    ///
    /// # Panics
    ///
    /// This method will panic if the length of the mapping does not match
    /// the length of the rear-coded list.
    pub fn from_parts(rcl: RearCodedList<I, O, D, P, SORTED>, map: Q) -> Self {
        assert_eq!(
            rcl.len(),
            map.len(),
            "Length mismatch between rear-coded list ({}) and mapping ({})",
            rcl.len(),
            map.len()
        );
        Self { rcl, map }
    }

    pub fn into_parts(self) -> (RearCodedList<I, O, D, P, SORTED>, Q) {
        (self.rcl, self.map)
    }

    /// Returns the number of elements.
    ///
    /// This method is equivalent to [`IndexedSeq::len`], but it is provided to
    /// reduce ambiguity in method resolution.
    #[inline]
    pub fn len(&self) -> usize {
        self.rcl.len()
    }

    /// Writes the index-th element to `result` as bytes. This is useful to avoid
    /// allocating a new vector for every query.
    #[inline]
    fn get_in_place_impl(&self, index: usize, result: &mut Vec<u8>) {
        let index = self.map.index_value(index);
        self.rcl.get_in_place_impl(index, result)
    }
}

impl<
    I: PartialEq<O> + PartialEq + ?Sized,
    O: PartialEq<I> + PartialEq,
    D: AsRef<[u8]>,
    P: AsRef<[usize]>,
    Q: SliceByValue<Value = usize>,
    const SORTED: bool,
> MappedRearCodedList<I, O, D, P, Q, SORTED>
where
    for<'a> Lend<'a, I, O, D, P, Q, SORTED>: Lender,
{
    /// Returns a [`Lender`] over the elements of the list.
    ///
    /// Note that [`iter`](RearCodedList::iter) is more convenient if
    /// you need owned elements.
    #[inline(always)]
    pub fn lender(&self) -> Lend<'_, I, O, D, P, Q, SORTED> {
        Lend::new(self)
    }

    /// Returns a [`Lender`] over the elements of the list
    /// starting from the given index.
    ///
    /// Note that [`iter`](RearCodedList::iter_from) is more convenient if
    /// you need owned elements.
    #[inline(always)]
    pub fn lender_from(&self, from: usize) -> Lend<'_, I, O, D, P, Q, SORTED> {
        Lend::new_from(self, from)
    }

    /// Returns an [`Iterator`] over the elements of the list.
    ///
    /// Note that [`lender`](RearCodedList::lender_from) is more efficient if
    /// you need to iterate over many elements.
    #[inline(always)]
    pub fn iter(&self) -> Iter<'_, I, O, D, P, Q, SORTED> {
        Iter(self.lender())
    }

    /// Returns an [`Iterator`] over the elements of the list
    /// starting from the given index.
    ///
    /// Note that [`lender`](RearCodedList::lender_from) is more efficient if
    /// you need to iterate over many elements.
    #[inline(always)]
    pub fn iter_from(&self, from: usize) -> Iter<'_, I, O, D, P, Q, SORTED> {
        Iter(self.lender_from(from))
    }
}

// Dictionary traits

impl<
    I: PartialEq<O> + PartialEq + ?Sized,
    O: PartialEq<I> + PartialEq,
    D: AsRef<[u8]>,
    P: AsRef<[usize]>,
    Q: SliceByValue<Value = usize>,
    const SORTED: bool,
> Types for MappedRearCodedList<I, O, D, P, Q, SORTED>
{
    type Output<'a> = O;
    type Input = I;
}

impl<D: AsRef<[u8]>, P: AsRef<[usize]>, Q: SliceByValue<Value = usize>, const SORTED: bool>
    IndexedSeq for MappedRearCodedList<[u8], Vec<u8>, D, P, Q, SORTED>
{
    #[inline(always)]
    unsafe fn get_unchecked(&self, index: usize) -> Self::Output<'_> {
        let index = self.map.index_value(index);
        self.rcl.get_unchecked(index)
    }

    #[inline(always)]
    fn len(&self) -> usize {
        self.rcl.len()
    }
}

impl<D: AsRef<[u8]>, P: AsRef<[usize]>, Q: SliceByValue<Value = usize>, const SORTED: bool>
    MappedRearCodedList<[u8], Vec<u8>, D, P, Q, SORTED>
{
    /// Returns in place the byte sequence of given index by writing
    /// its bytes into the provided vector.
    pub fn get_in_place(&self, index: usize, result: &mut Vec<u8>) {
        let index = self.map.index_value(index);
        self.rcl.get_in_place_impl(index, result);
    }
}

impl<D: AsRef<[u8]>, P: AsRef<[usize]>, Q: SliceByValue<Value = usize>, const SORTED: bool>
    IndexedSeq for MappedRearCodedList<str, String, D, P, Q, SORTED>
{
    #[inline(always)]
    unsafe fn get_unchecked(&self, index: usize) -> Self::Output<'_> {
        let index = self.map.index_value(index);
        self.rcl.get_unchecked(index)
    }

    #[inline(always)]
    fn len(&self) -> usize {
        self.rcl.len()
    }
}

impl<D: AsRef<[u8]>, P: AsRef<[usize]>, Q: SliceByValue<Value = usize>, const SORTED: bool>
    MappedRearCodedList<str, String, D, P, Q, SORTED>
{
    /// Returns in place the string of given index by writing
    /// its bytes into the provided string.
    pub fn get_in_place(&self, index: usize, result: &mut String) {
        let index = self.map.index_value(index);
        self.rcl.get_in_place(index, result)
    }

    /// Returns the bytes of the string of given index.
    ///
    /// This method can be used to avoid UTF-8 checks when you just need the raw
    /// bytes, or to use methods such as [`String::from_utf8_unchecked`] and
    /// [`str::from_utf8_unchecked`] to avoid the cost UTF-8 checks. Be aware,
    /// however, that using invalid UTF-8 data may lead to undefined behavior.
    #[inline]
    pub fn get_bytes(&self, index: usize) -> Vec<u8> {
        let index = self.map.index_value(index);
        self.rcl.get_bytes(index)
    }

    /// Returns in place the string of given index by writing
    /// its bytes into the provided vector.
    ///
    /// This method can be used to avoid UTF-8 checks when you just need the raw
    /// bytes, or to use methods such as [`String::from_utf8_unchecked`] and
    /// [`str::from_utf8_unchecked`] to avoid the cost UTF-8 checks. Be aware,
    /// however, that using invalid UTF-8 data may lead to undefined behavior.
    #[inline(always)]
    pub fn get_bytes_in_place(&self, index: usize, result: &mut Vec<u8>) {
        let index = self.map.index_value(index);
        self.rcl.get_in_place_impl(index, result);
    }
}

// Lenders

/// Sequential [`Lender`] over the contents of the list.
#[derive(Debug, Clone, MemDbg, MemSize)]
pub struct Lend<
    'a,
    I: PartialEq<O> + PartialEq + ?Sized,
    O: PartialEq<I> + PartialEq,
    D: AsRef<[u8]>,
    P: AsRef<[usize]>,
    Q: SliceByValue<Value = usize>,
    const SORTED: bool,
> {
    prcl: &'a MappedRearCodedList<I, O, D, P, Q, SORTED>,
    buffer: Vec<u8>,
    index: usize,
}

impl<
    'a,
    I: PartialEq<O> + PartialEq + ?Sized,
    O: PartialEq<I> + PartialEq,
    D: AsRef<[u8]>,
    P: AsRef<[usize]>,
    Q: SliceByValue<Value = usize>,
    const SORTED: bool,
> Lend<'a, I, O, D, P, Q, SORTED>
where
    Self: Lender,
{
    /// Creates a new lender over the rear-coded list.
    pub fn new(prcl: &'a MappedRearCodedList<I, O, D, P, Q, SORTED>) -> Self {
        Self {
            prcl,
            buffer: Vec::with_capacity(128),
            index: 0,
        }
    }

    /// Creates a new lender over the rear-coded list starting from the given
    /// position.
    pub fn new_from(prcl: &'a MappedRearCodedList<I, O, D, P, Q, SORTED>, from: usize) -> Self {
        Self {
            prcl,
            buffer: Vec::with_capacity(128),
            index: from,
        }
    }

    /// Internal next method that returns a reference to the inner buffer.
    fn next_impl(&mut self) -> Option<&[u8]> {
        if self.index >= self.prcl.len() {
            return None;
        }
        self.prcl.get_in_place_impl(self.index, &mut self.buffer);
        self.index += 1;
        Some(&self.buffer)
    }
}

impl<
    'a,
    'b,
    I: PartialEq<O> + PartialEq + ?Sized,
    O: PartialEq<I> + PartialEq,
    D: AsRef<[u8]>,
    P: AsRef<[usize]>,
    Q: SliceByValue<Value = usize>,
    const SORTED: bool,
> Lending<'b> for Lend<'a, I, O, D, P, Q, SORTED>
{
    type Lend = &'b I;
}

impl<
    O: PartialEq<str> + PartialEq,
    D: AsRef<[u8]>,
    P: AsRef<[usize]>,
    Q: SliceByValue<Value = usize>,
    const SORTED: bool,
> Lender for Lend<'_, str, O, D, P, Q, SORTED>
where
    str: PartialEq<O> + PartialEq,
{
    fn next(&mut self) -> Option<&'_ str> {
        self.next_impl().map(|s| std::str::from_utf8(s).unwrap())
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }
}

impl<
    O: PartialEq<[u8]> + PartialEq,
    D: AsRef<[u8]>,
    P: AsRef<[usize]>,
    Q: SliceByValue<Value = usize>,
    const SORTED: bool,
> Lender for Lend<'_, [u8], O, D, P, Q, SORTED>
where
    [u8]: PartialEq<O> + PartialEq,
{
    fn next(&mut self) -> Option<&[u8]> {
        self.next_impl()
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }
}

impl<
    'a,
    I: PartialEq<O> + PartialEq + ?Sized,
    O: PartialEq<I> + PartialEq,
    D: AsRef<[u8]>,
    P: AsRef<[usize]>,
    Q: SliceByValue<Value = usize>,
    const SORTED: bool,
> ExactSizeLender for Lend<'a, I, O, D, P, Q, SORTED>
where
    Lend<'a, I, O, D, P, Q, SORTED>: Lender,
{
    #[inline(always)]
    fn len(&self) -> usize {
        self.prcl.len() - self.index
    }
}

impl<
    'a,
    I: PartialEq<O> + PartialEq + ?Sized,
    O: PartialEq<I> + PartialEq,
    D: AsRef<[u8]>,
    P: AsRef<[usize]>,
    Q: SliceByValue<Value = usize>,
    const SORTED: bool,
> FusedLender for Lend<'a, I, O, D, P, Q, SORTED>
where
    Lend<'a, I, O, D, P, Q, SORTED>: Lender,
{
}

// Iterators

/// Sequential [`Iterator`] over the the contents of the list.
#[derive(Debug, Clone, MemDbg, MemSize)]
pub struct Iter<
    'a,
    I: PartialEq<O> + PartialEq + ?Sized,
    O: PartialEq<I> + PartialEq,
    D: AsRef<[u8]>,
    P: AsRef<[usize]>,
    Q: SliceByValue<Value = usize>,
    const SORTED: bool,
>(Lend<'a, I, O, D, P, Q, SORTED>);

impl<
    'a,
    I: PartialEq<O> + PartialEq + ?Sized,
    O: PartialEq<I> + PartialEq,
    D: AsRef<[u8]>,
    P: AsRef<[usize]>,
    Q: SliceByValue<Value = usize>,
    const SORTED: bool,
> std::iter::ExactSizeIterator for Iter<'a, I, O, D, P, Q, SORTED>
where
    Iter<'a, I, O, D, P, Q, SORTED>: std::iter::Iterator,
    Lend<'a, I, O, D, P, Q, SORTED>: ExactSizeLender,
{
    #[inline(always)]
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl<
    'a,
    I: PartialEq<O> + PartialEq + ?Sized,
    O: PartialEq<I> + PartialEq,
    D: AsRef<[u8]>,
    P: AsRef<[usize]>,
    Q: SliceByValue<Value = usize>,
    const SORTED: bool,
> std::iter::FusedIterator for Iter<'a, I, O, D, P, Q, SORTED>
where
    Iter<'a, I, O, D, P, Q, SORTED>: std::iter::Iterator,
    Lend<'a, str, String, D, P, Q, SORTED>: FusedLender,
{
}

impl<'a, D: AsRef<[u8]>, P: AsRef<[usize]>, Q: SliceByValue<Value = usize>, const SORTED: bool>
    std::iter::Iterator for Iter<'a, str, String, D, P, Q, SORTED>
where
    Lend<'a, str, String, D, P, Q, SORTED>: Lender,
{
    type Item = String;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        // SAFETY: We encoded valid UTF-8 strings
        self.0
            .next_impl()
            .map(|v| String::from_utf8(Vec::from(v)).unwrap())
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }
}

impl<'a, D: AsRef<[u8]>, P: AsRef<[usize]>, Q: SliceByValue<Value = usize>, const SORTED: bool>
    std::iter::Iterator for Iter<'a, [u8], Vec<u8>, D, P, Q, SORTED>
where
    Lend<'a, [u8], Vec<u8>, D, P, Q, SORTED>: ExactSizeLender,
{
    type Item = Vec<u8>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        // SAFETY: We encoded valid UTF-8 strings
        self.0.next_impl().map(|v| v.into())
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }
}

// Into... impls

impl<
    'a,
    I: PartialEq<O> + PartialEq + ?Sized,
    O: PartialEq<I> + PartialEq,
    D: AsRef<[u8]>,
    P: AsRef<[usize]>,
    Q: SliceByValue<Value = usize>,
    const SORTED: bool,
> IntoLender for &'a MappedRearCodedList<I, O, D, P, Q, SORTED>
where
    Lend<'a, I, O, D, P, Q, SORTED>: Lender,
{
    type Lender = Lend<'a, I, O, D, P, Q, SORTED>;
    #[inline(always)]
    fn into_lender(self) -> Lend<'a, I, O, D, P, Q, SORTED> {
        Lend::new(self)
    }
}

impl<
    'a,
    I: PartialEq<O> + PartialEq + ?Sized,
    O: PartialEq<I> + PartialEq,
    D: AsRef<[u8]>,
    P: AsRef<[usize]>,
    Q: SliceByValue<Value = usize>,
    const SORTED: bool,
> IntoIterator for &'a MappedRearCodedList<I, O, D, P, Q, SORTED>
where
    for<'b> Lend<'b, I, O, D, P, Q, SORTED>: Lender,
    Iter<'a, I, O, D, P, Q, SORTED>: std::iter::Iterator,
{
    type Item = <Iter<'a, I, O, D, P, Q, SORTED> as Iterator>::Item;
    type IntoIter = Iter<'a, I, O, D, P, Q, SORTED>;
    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<
    'a,
    I: PartialEq<O> + PartialEq + ?Sized,
    O: PartialEq<I> + PartialEq,
    D: AsRef<[u8]>,
    P: AsRef<[usize]>,
    Q: SliceByValue<Value = usize>,
    const SORTED: bool,
> IntoIteratorFrom for &'a MappedRearCodedList<I, O, D, P, Q, SORTED>
where
    for<'b> Lend<'b, I, O, D, P, Q, SORTED>: Lender,
    Iter<'a, I, O, D, P, Q, SORTED>: std::iter::Iterator,
{
    type IntoIterFrom = Iter<'a, I, O, D, P, Q, SORTED>;
    #[inline(always)]
    fn into_iter_from(self, from: usize) -> Self::IntoIter {
        self.iter_from(from)
    }
}
