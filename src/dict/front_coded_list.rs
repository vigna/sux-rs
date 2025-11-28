/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Tommaso Fontana
 * SPDX-FileCopyrightText: 2025 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Compressed strings (`str`) and byte sequences (`[u8]`) immutable storage by
//! front-coded prefix omission.
//!
//! Prefix omission compresses a list of sequences omitting the common prefixes
//! of consecutive sequences. To do so, it stores the length of the common
//! prefix (hence, front coding). It is usually applied to lists of elements
//! sorted in ascending order. The elements can contain arbitrary data,
//! including `\0` bytes. Note that if your list is not sorted and you want to
//! achieve significant compression you can try to use a
//! [`MappedRearCodedList`](crate::dict::mapped_rear_coded_list) instead.
//!
//! The encoding is done in blocks of `ratio` elements: in each block the first
//! string is encoded without compression, whereas the other elements are encoded
//! with the common prefix removed.
//!
//! The structure is parameterized by the input/output type and by the storage
//! backend, but two versions are presently implemented, and associated with
//! a type alias:
//!
//! - [`FrontCodedListStr`]: stores UTF-8 elements (`AsRef<str>`), returning
//!   [`String`] on access;
//!
//! - [`FrontCodedListSliceU8`]: stores byte sequences (`AsRef<[u8]>`), returning
//!   `Vec<u8>` on access.
//!
//! Besides the standard access by means of the [`IndexedSeq`] trait, this
//! structure also implements the `get_in_place` method, which allows to write
//! the element directly into a user-provided buffer (a string or a vector of
//! bytes), avoiding allocations. [`FrontCodedListStr`] has an additional
//! [`get_bytes_in_place`](FrontCodedListStr::get_bytes_in_place) method that
//! writes the bytes of the string into a user-provided `Vec<u8>`.
//!
//! Front-coded lists can be iterated upon using either an
//! [`Iterator`](FrontCodedList::iter) or a [`Lender`](FrontCodedList::lender).
//! In the first case there will be an allocation at each iteration, whereas in
//! second case a single buffer will be reused. You can also
//! [iterate from a given position](FrontCodedList::lender_from), which is
//! much faster than skipping elements one by one.
//!
//! There are two versions of the structure, depending on a const boolean
//! parameter `SORTED`: if it is true, the elements must be sorted in ascending
//! order (this is the usual case for obtaining good compression), and an
//! implementation of [`IndexedDict`] will be available that finds the position
//! of elements in the list by binary search; if false, the elements can be in
//! arbitrary order, but no dictionary operations will be available (this
//! configuration is mainly useful for storing large lists of elements with quick
//! deserialization or easy memory mapping).
//!
//! To build a [`FrontCodedList`] you use a [`FrontCodedListBuilder`], which has a
//! first parameter, `str` or `[u8]`, that specifies the type of elements, and a
//! `SORTED` boolean parameter; you have to [`push`](FrontCodedListBuilder::push)
//! the elements, and then call the [`build`](FrontCodedListBuilder::build)
//! method to obtain the final structure.
//!
//! If the feature `epserde` is active, you can also directly serialize a
//! front-coded list without building it in memory using the functions
//! [`serialize_str`] / [`serialize_slice_u8`] / [`store_str`] /
//! [`store_slice_u8`]. They use an advanced feature of ε-serde that makes it
//! possible to serialize iterators and deserialize vectors or boxed slices. The
//! method is about three times slower than using a [`FrontCodedListBuilder`],
//! but it uses very little memory. Since you can memory map an instance of this
//! class with ε-serde, this allows to create and use lists that would not fit
//! into memory.
//!
//! # UTF-8 Encoding and Panics
//!
//! [`FrontCodedListStr`] methods may panic if the stored data is not valid UTF-8.
//! That can happen only in case of data corruption (e.g., on serialized data)
//! or bugs in the construction code.
//!
//! The check for UTF-8 validity poses a significant performance penalty on the
//! accessors. If you are sure that the data is valid UTF-8 (e.g., you built the
//! serialized structure yourself), you can use
//! [`get_bytes`](FrontCodedListStr::get_bytes) or
//! [`get_bytes_in_place`](FrontCodedListStr::get_bytes_in_place) and then unsafe
//! methods such as [`String::from_utf8_unchecked`] or
//! [`str::from_utf8_unchecked`].
//!
//! # Examples
//!
//! Here we use a [`FrontCodedListBuilder`] to build a sorted front-coded list:
//!
//! ```
//! use sux::traits::{IndexedSeq, IndexedDict};
//! use sux::dict::FrontCodedListBuilder;
//!
//! let mut fclb = FrontCodedListBuilder::<str, true>::new(4);
//!
//! fclb.push("aa");
//! fclb.push("aab");
//! fclb.push("abc");
//! fclb.push("abdd");
//! fclb.push("abde\0f");
//! fclb.push("abdf");
//!
//! let fcl = fclb.build();
//! assert_eq!(fcl.len(), 6);
//! assert_eq!(fcl.get(0), "aa");
//! assert_eq!(fcl.get(1), "aab");
//! assert_eq!(fcl.get(2), "abc");
//! assert_eq!(fcl.index_of("abde\0f"), Some(4));
//! assert_eq!(fcl.index_of("foo"), None);
//! ```
//!
//! Here instead we serialize directly the list in an aligned cursor. Note that
//! the methods accepts a
//! [`FallibleRewindableLender`](crate::utils::lenders::FallibleRewindableLender),
//! so we create it from a buffer using the
//! [`FromSlice`](crate::utils::FromSlice) adapter. Using the [`store_str`]
//! function you could write directly to a file.
//!
//! ```
//! # #[cfg(feature = "epserde")] {
//! use std::io::{Cursor, Write};
//! use sux::traits::{IndexedSeq, IndexedDict};
//! use sux::dict::FrontCodedListStr;
//! use sux::dict::front_coded_list;
//! use sux::utils::LineLender;
//! use sux::utils::FromSlice;
//! use maligned::A16;
//! use epserde::prelude::*;
//!
//! let strings = vec!["aa", "aab", "abc", "abdd", "abde\0f", "abdf"];
//! let mut cursor = <AlignedCursor<A16>>::new();
//! front_coded_list::serialize_str::<_, _, true>(4, FromSlice::new(&strings), &mut cursor)?;
//! cursor.set_position(0);
//! let fcl = unsafe {
//!    <FrontCodedListStr<true>>::deserialize_full(&mut cursor)?
//! };
//! assert_eq!(fcl.len(), 6);
//! assert_eq!(fcl.get(0), "aa");
//! assert_eq!(fcl.get(1), "aab");
//! assert_eq!(fcl.get(2), "abc");
//! assert_eq!(fcl.index_of("abde\0f"), Some(4));
//! assert_eq!(fcl.index_of("foo"), None);
//! # }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Format
//!
//! The front-coded list keeps an array of pointers to the beginning of each block
//! in data. Due do the omission of prefixes, we can only start decoding from
//! the beginning of a block as we write a full string there.
//!
//! The `data` portion contains strings in the following format:
//! - if the index of the string is a multiple of `ratio`, we write: `<len><bytes>`
//! - otherwise, we write: `<suffix_len><lcp><suffix_bytes>`
//!
//! where `<len>`, `<suffix_len>`, and `<lcp>` are VByte-encoded integers
//! and `<bytes>` and `<suffix_bytes>` are the actual bytes of the string or suffix.

use crate::traits::{IndexedDict, IndexedSeq, IntoIteratorFrom, Types};
use core::marker::PhantomData;
use lender::{ExactSizeLender, IntoLender, Lender, Lending};
use lender::{FusedLender, for_};
use mem_dbg::*;
use std::borrow::Borrow;

#[derive(Debug, Clone, MemDbg, MemSize, Default)]
/// Statistics of the encoded data.
struct Stats {
    /// Maximum block size in bytes.
    pub max_block_bytes: usize,
    /// The total sum of the block size in bytes.
    pub sum_block_bytes: usize,

    /// Maximum shared prefix in bytes.
    pub max_lcp: usize,
    /// The total sum of the shared prefix in bytes.
    pub sum_lcp: usize,

    /// Maximum string length in bytes.
    pub max_str_len: usize,
    /// The total sum of the string lengths in bytes.
    pub sum_str_len: usize,

    /// The number of bytes used to store the lcp lengths in data.
    pub code_bytes: usize,
    /// The number of bytes used to store the suffixes in data.
    pub suffixes_bytes: usize,

    /// The bytes wasted writing without compression the first string in block.
    pub redundancy: isize,
}

/// Main structure; please use the type aliases [`FrontCodedListStr`] and
/// [`FrontCodedListSliceU8`].
///
/// The parameters `I` and `O` are the input and output types, respectively, and
/// they are used in the [`IndexedSeq`] implementation. The parameters `D` and
/// `P` are the storage backends for the encoded data and the block pointers,
/// respectively. The const boolean parameter `SORTED` specifies whether the
/// elements are sorted in ascending order and enables, if true, the
/// implementation of [`IndexedDict`].
#[derive(Debug, Clone, MemDbg, MemSize)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FrontCodedList<I: ?Sized, O, D = Box<[u8]>, P = Box<[usize]>, const SORTED: bool = true> {
    /// The number of strings in a block; this value trades off compression for speed.
    ratio: usize,
    /// Number of encoded strings.
    len: usize,
    /// The encoded strings, `\0`-terminated.
    data: D,
    /// The pointer to the starting string of each block.
    pointers: P,
    _marker_i: PhantomData<I>,
    _marker_o: PhantomData<O>,
}

/// A front-coded list of byte sequences.
pub type FrontCodedListSliceU8<const SORTED: bool = true> =
    FrontCodedList<[u8], Vec<u8>, Box<[u8]>, Box<[usize]>, SORTED>;
/// A front-coded list of strings.
pub type FrontCodedListStr<const SORTED: bool = true> =
    FrontCodedList<str, String, Box<[u8]>, Box<[usize]>, SORTED>;

impl<
    I: PartialEq<O> + PartialEq + ?Sized,
    O: PartialEq<I> + PartialEq,
    D: AsRef<[u8]>,
    P: AsRef<[usize]>,
    const SORTED: bool,
> FrontCodedList<I, O, D, P, SORTED>
{
    /// Returns the number of elements.
    ///
    /// This method is equivalent to [`IndexedSeq::len`], but it is provided to
    /// reduce ambiguity in method resolution.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Writes the index-th element to `result` as bytes. This is useful to avoid
    /// allocating a new vector for every query.
    #[inline]
    pub(super) fn get_in_place_impl(&self, index: usize, result: &mut Vec<u8>) {
        result.clear();
        let block = index / self.ratio;
        let offset = index % self.ratio;

        let start = self.pointers.as_ref()[block];
        let mut data = &self.data.as_ref()[start..];

        // declare these vars so the decoding looks cleaner
        let (mut lcp, mut len);
        // decode the first string in the block
        (len, data) = decode_int(data);
        result.extend_from_slice(&data[..len]);
        data = &data[len..];

        for _ in 0..offset {
            // get the suffix length and the lcp
            (len, data) = decode_int(data);
            (lcp, data) = decode_int(data);
            // truncate to keep only the common prefix
            result.truncate(lcp);
            // copy the new suffix
            result.extend_from_slice(&data[..len]);
            data = &data[len..];
        }
    }
}

impl<
    I: PartialEq<O> + PartialEq + ?Sized,
    O: PartialEq<I> + PartialEq,
    D: AsRef<[u8]>,
    P: AsRef<[usize]>,
    const SORTED: bool,
> FrontCodedList<I, O, D, P, SORTED>
where
    for<'a> Lend<'a, I, O, D, P, SORTED>: Lender,
{
    /// Returns a [`Lender`] over the elements of the list.
    ///
    /// Note that [`iter`](FrontCodedList::iter) is more convenient if
    /// you need owned elements.
    #[inline(always)]
    pub fn lender(&self) -> Lend<'_, I, O, D, P, SORTED> {
        Lend::new(self)
    }

    /// Returns a [`Lender`] over the elements of the list
    /// starting from the given index.
    ///
    /// Note that [`iter`](FrontCodedList::iter_from) is more convenient if
    /// you need owned elements.
    #[inline(always)]
    pub fn lender_from(&self, from: usize) -> Lend<'_, I, O, D, P, SORTED> {
        Lend::new_from(self, from)
    }

    /// Returns an [`Iterator`] over the elements of the list.
    ///
    /// Note that [`lender`](FrontCodedList::lender_from) is more efficient if
    /// you need to iterate over many elements.
    #[inline(always)]
    pub fn iter(&self) -> Iter<'_, I, O, D, P, SORTED> {
        Iter(self.lender())
    }

    /// Returns an [`Iterator`] over the elements of the list
    /// starting from the given index.
    ///
    /// Note that [`lender`](FrontCodedList::lender_from) is more efficient if
    /// you need to iterate over many elements.
    #[inline(always)]
    pub fn iter_from(&self, from: usize) -> Iter<'_, I, O, D, P, SORTED> {
        Iter(self.lender_from(from))
    }
}

impl<I: ?Sized, O, D: AsRef<[u8]>, P: AsRef<[usize]>> FrontCodedList<I, O, D, P, true> {
    /// Internal method that finds the index of a slice of bytes using a binary
    /// search on the blocks followed by a linear search within the block.
    ///
    /// Note that we use Rust built-in binary search: thus, in case of multiple
    /// occurrences of the same string, we may return any of them.
    ///
    /// Use by the implementation of [`IndexedDict::index_of`].
    fn index_of(&self, value: impl AsRef<[u8]>) -> Option<usize> {
        let string = value.as_ref();
        // first to a binary search on the blocks to find the block
        let block_idx = self.pointers.as_ref().binary_search_by(|block_ptr| {
            // do a quick in-place decode of the explicit string at the beginning of the block
            let data = &self.data.as_ref()[*block_ptr..];
            let (len, tmp) = decode_int(data);
            memcmp(string, &tmp[..len]).reverse()
        });

        if let Ok(block_idx) = block_idx {
            return Some(block_idx * self.ratio);
        }

        let mut block_idx = block_idx.unwrap_err();
        if block_idx == 0 || block_idx > self.pointers.as_ref().len() {
            // the string is before the first block
            return None;
        }
        block_idx -= 1;
        // finish by a linear search on the block
        let mut result = Vec::with_capacity(128);
        let start = self.pointers.as_ref()[block_idx];
        let mut data = &self.data.as_ref()[start..];

        let (mut to_copy, mut lcp);
        // decode the first string in the block
        (to_copy, data) = decode_int(data);
        result.extend_from_slice(&data[..to_copy]);
        data = &data[to_copy..];

        let in_block = (self.ratio - 1).min(self.len - block_idx * self.ratio - 1);
        for idx in 0..in_block {
            // get the suffix length and the lcp
            (to_copy, data) = decode_int(data);
            (lcp, data) = decode_int(data);
            // truncate to keep only the common prefix
            result.truncate(lcp);
            // copy the new suffix
            result.extend_from_slice(&data[..to_copy]);
            data = &data[to_copy..];
            // TODO!: this can be optimized to avoid the copy
            match memcmp_rust(string, &result) {
                core::cmp::Ordering::Greater => {}
                core::cmp::Ordering::Equal => return Some(block_idx * self.ratio + idx + 1),
                core::cmp::Ordering::Less => return None,
            }
        }
        None
    }
}

// Dictionary traits

impl<
    I: PartialEq<O> + PartialEq + ?Sized,
    O: PartialEq<I> + PartialEq,
    D: AsRef<[u8]>,
    P: AsRef<[usize]>,
    const SORTED: bool,
> Types for FrontCodedList<I, O, D, P, SORTED>
{
    type Output<'a> = O;
    type Input = I;
}

impl<D: AsRef<[u8]>, P: AsRef<[usize]>, const SORTED: bool> IndexedSeq
    for FrontCodedList<[u8], Vec<u8>, D, P, SORTED>
{
    #[inline(always)]
    unsafe fn get_unchecked(&self, index: usize) -> Self::Output<'_> {
        let mut result = Vec::with_capacity(128);
        self.get_in_place_impl(index, &mut result);
        result
    }

    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }
}

impl<D: AsRef<[u8]>, P: AsRef<[usize]>, const SORTED: bool>
    FrontCodedList<[u8], Vec<u8>, D, P, SORTED>
{
    /// Returns in place the byte sequence of given index by writing
    /// its bytes into the provided vector.
    pub fn get_in_place(&self, index: usize, result: &mut Vec<u8>) {
        self.get_in_place_impl(index, result);
    }
}

impl<D: AsRef<[u8]>, P: AsRef<[usize]>, const SORTED: bool> IndexedSeq
    for FrontCodedList<str, String, D, P, SORTED>
{
    #[inline(always)]
    unsafe fn get_unchecked(&self, index: usize) -> Self::Output<'_> {
        let mut result = Vec::with_capacity(128);
        self.get_in_place_impl(index, &mut result);
        // SAFETY: we encoded valid UTF-8 strings
        debug_assert!(std::str::from_utf8(&result).is_ok());
        String::from_utf8(result).unwrap()
    }

    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }
}

impl<D: AsRef<[u8]>, P: AsRef<[usize]>, const SORTED: bool>
    FrontCodedList<str, String, D, P, SORTED>
{
    /// Returns in place the string of given index by writing
    /// its bytes into the provided string.
    pub fn get_in_place(&self, index: usize, result: &mut String) {
        let mut buffer = Vec::with_capacity(64);
        self.get_in_place_impl(index, &mut buffer);
        result.clear();
        debug_assert!(std::str::from_utf8(&buffer).is_ok());
        result.push_str(std::str::from_utf8(&buffer).unwrap());
    }

    /// Returns the bytes of the string of given index.
    ///
    /// This method can be used to avoid UTF-8 checks when you just need the raw
    /// bytes, or to use methods such as [`String::from_utf8_unchecked`] and
    /// [`str::from_utf8_unchecked`] to avoid the cost UTF-8 checks. Be aware,
    /// however, that using invalid UTF-8 data may lead to undefined behavior.
    #[inline]
    pub fn get_bytes(&self, index: usize) -> Vec<u8> {
        let mut buf = Vec::with_capacity(64);
        self.get_in_place_impl(index, &mut buf);
        buf
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
        self.get_in_place_impl(index, result);
    }
}

/// Implementation of [`IndexedDict`] for sorted front-coded lists.
///
/// Note that we use Rust built-in binary search: thus, in case of multiple
/// occurrences of the same element, we may return any of them.
impl<
    I: PartialEq<O> + PartialEq + ?Sized + AsRef<[u8]>,
    O: PartialEq<I> + PartialEq,
    D: AsRef<[u8]>,
    P: AsRef<[usize]>,
> IndexedDict for FrontCodedList<I, O, D, P, true>
{
    fn index_of(&self, value: impl Borrow<Self::Input>) -> Option<usize> {
        self.index_of(value.borrow())
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
    const SORTED: bool,
> {
    fcl: &'a FrontCodedList<I, O, D, P, SORTED>,
    data: &'a [u8],
    buffer: Vec<u8>,
    index: usize,
}

impl<
    'a,
    I: PartialEq<O> + PartialEq + ?Sized,
    O: PartialEq<I> + PartialEq,
    D: AsRef<[u8]>,
    P: AsRef<[usize]>,
    const SORTED: bool,
> Lend<'a, I, O, D, P, SORTED>
where
    Self: Lender,
{
    /// Creates a new lender over the front-coded list.
    pub fn new(fcl: &'a FrontCodedList<I, O, D, P, SORTED>) -> Self {
        Self {
            fcl,
            data: fcl.data.as_ref(),
            buffer: Vec::with_capacity(128),
            index: 0,
        }
    }

    /// Creates a new lender over the front-coded list starting from the given
    /// position.
    pub fn new_from(fcl: &'a FrontCodedList<I, O, D, P, SORTED>, from: usize) -> Self {
        let block = from / fcl.ratio;
        let offset = from % fcl.ratio;
        let start = fcl.pointers.as_ref()[block];
        let mut res = Lend {
            fcl,
            index: block * fcl.ratio,
            data: &fcl.data.as_ref()[start..],
            buffer: Vec::with_capacity(128),
        };
        for _ in 0..offset {
            res.next();
        }
        res
    }

    /// Internal next method that returns a reference to the inner buffer.
    fn next_impl(&mut self) -> Option<&[u8]> {
        if self.index >= self.fcl.len() {
            return None;
        }

        // figure out how much of the suffix we have to read
        let (to_copy, mut tmp) = decode_int(self.data);
        // figure out how much of the buffer we have to keep
        let lcp = if self.index % self.fcl.ratio == 0 {
            0
        } else {
            let lcp;
            (lcp, tmp) = decode_int(tmp);
            lcp
        };
        // truncate the buffer to keep only the relevant part
        self.buffer.truncate(lcp);
        // copy the new suffix
        self.buffer.extend_from_slice(&tmp[..to_copy]);
        self.data = &tmp[to_copy..];
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
    const SORTED: bool,
> Lending<'b> for Lend<'a, I, O, D, P, SORTED>
{
    type Lend = &'b I;
}

impl<O: PartialEq<str> + PartialEq, D: AsRef<[u8]>, P: AsRef<[usize]>, const SORTED: bool> Lender
    for Lend<'_, str, O, D, P, SORTED>
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

impl<O: PartialEq<[u8]> + PartialEq, D: AsRef<[u8]>, P: AsRef<[usize]>, const SORTED: bool> Lender
    for Lend<'_, [u8], O, D, P, SORTED>
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
    const SORTED: bool,
> ExactSizeLender for Lend<'a, I, O, D, P, SORTED>
where
    Lend<'a, I, O, D, P, SORTED>: Lender,
{
    #[inline(always)]
    fn len(&self) -> usize {
        self.fcl.len() - self.index
    }
}

impl<
    'a,
    I: PartialEq<O> + PartialEq + ?Sized,
    O: PartialEq<I> + PartialEq,
    D: AsRef<[u8]>,
    P: AsRef<[usize]>,
    const SORTED: bool,
> FusedLender for Lend<'a, I, O, D, P, SORTED>
where
    Lend<'a, I, O, D, P, SORTED>: Lender,
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
    const SORTED: bool,
>(Lend<'a, I, O, D, P, SORTED>);

impl<
    'a,
    I: PartialEq<O> + PartialEq + ?Sized,
    O: PartialEq<I> + PartialEq,
    D: AsRef<[u8]>,
    P: AsRef<[usize]>,
    const SORTED: bool,
> std::iter::ExactSizeIterator for Iter<'a, I, O, D, P, SORTED>
where
    Iter<'a, I, O, D, P, SORTED>: std::iter::Iterator,
    Lend<'a, I, O, D, P, SORTED>: ExactSizeLender,
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
    const SORTED: bool,
> std::iter::FusedIterator for Iter<'a, I, O, D, P, SORTED>
where
    Iter<'a, I, O, D, P, SORTED>: std::iter::Iterator,
    Lend<'a, str, String, D, P, SORTED>: FusedLender,
{
}

impl<'a, D: AsRef<[u8]>, P: AsRef<[usize]>, const SORTED: bool> std::iter::Iterator
    for Iter<'a, str, String, D, P, SORTED>
where
    Lend<'a, str, String, D, P, SORTED>: Lender,
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

impl<'a, D: AsRef<[u8]>, P: AsRef<[usize]>, const SORTED: bool> std::iter::Iterator
    for Iter<'a, [u8], Vec<u8>, D, P, SORTED>
where
    Lend<'a, [u8], Vec<u8>, D, P, SORTED>: ExactSizeLender,
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
    const SORTED: bool,
> IntoLender for &'a FrontCodedList<I, O, D, P, SORTED>
where
    Lend<'a, I, O, D, P, SORTED>: Lender,
{
    type Lender = Lend<'a, I, O, D, P, SORTED>;
    #[inline(always)]
    fn into_lender(self) -> Lend<'a, I, O, D, P, SORTED> {
        Lend::new(self)
    }
}

impl<
    'a,
    I: PartialEq<O> + PartialEq + ?Sized,
    O: PartialEq<I> + PartialEq,
    D: AsRef<[u8]>,
    P: AsRef<[usize]>,
    const SORTED: bool,
> IntoIterator for &'a FrontCodedList<I, O, D, P, SORTED>
where
    for<'b> Lend<'b, I, O, D, P, SORTED>: Lender,
    Iter<'a, I, O, D, P, SORTED>: std::iter::Iterator,
{
    type Item = <Iter<'a, I, O, D, P, SORTED> as Iterator>::Item;
    type IntoIter = Iter<'a, I, O, D, P, SORTED>;
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
    const SORTED: bool,
> IntoIteratorFrom for &'a FrontCodedList<I, O, D, P, SORTED>
where
    for<'b> Lend<'b, I, O, D, P, SORTED>: Lender,
    Iter<'a, I, O, D, P, SORTED>: std::iter::Iterator,
{
    type IntoIterFrom = Iter<'a, I, O, D, P, SORTED>;
    #[inline(always)]
    fn into_iter_from(self, from: usize) -> Self::IntoIter {
        self.iter_from(from)
    }
}

/// Builder for a front-coded list.
///
/// You have to specify two types: the input type (currently, either `str` or `[u8]`)
/// and a boolean constant `SORTED` that indicates whether the strings are sorted
/// in ascending order (which will enable checks and indexing capabilities).
#[derive(Debug, Clone, MemDbg, MemSize)]
pub struct FrontCodedListBuilder<I: ?Sized, const SORTED: bool = true> {
    /// The number of strings in a block; this value trades compression for speed.
    ratio: usize,
    /// Number of encoded strings.
    len: usize,
    /// The encoded strings, `\0`-terminated.
    data: Vec<u8>,
    /// The total number of bytes written; this can be different
    /// than the length of data in the low-memory construction,
    /// as we truncate data after each push
    written_bytes: usize,
    /// The pointer to the starting string of each block.
    pointers: Vec<usize>,
    /// Statistics of the encoded data.
    stats: Stats,
    /// Cache of the last encoded string for incremental encoding.
    last_str: Vec<u8>,
    _marker: PhantomData<I>,
}

#[inline(always)]
/// Like memcmp
fn memcmp(string: &[u8], data: &[u8]) -> core::cmp::Ordering {
    for (a, b) in string.iter().zip(data.iter()) {
        let ord = a.cmp(b);
        if ord != core::cmp::Ordering::Equal {
            return ord;
        }
    }
    string.len().cmp(&data.len())
}

#[inline(always)]
/// Like memcmp, but both string are Rust strings.
fn memcmp_rust(string: &[u8], other: &[u8]) -> core::cmp::Ordering {
    for (a, b) in string.iter().zip(other.iter()) {
        let ord = a.cmp(b);
        if ord != core::cmp::Ordering::Equal {
            return ord;
        }
    }
    string.len().cmp(&other.len())
}

impl<I: ?Sized, const SORTED: bool> FrontCodedListBuilder<I, SORTED> {
    /// Creates a builder for a front-coded list with a block size of `ratio`.
    pub fn new(ratio: usize) -> Self {
        Self {
            data: Vec::with_capacity(1024),
            last_str: Vec::with_capacity(1024),
            pointers: Vec::new(),
            len: 0,
            written_bytes: 0,
            ratio,
            stats: Stats::default(),
            _marker: PhantomData,
        }
    }

    /// Returns the number of strings in the list.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns whether the builder is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Prints in a human-readable format the statistics of the
    /// strings currently in the builder.
    pub fn print_stats(&self) {
        println!(
            "{:>20}: {:>10}",
            "max_block_bytes", self.stats.max_block_bytes
        );
        println!(
            "{:>20}: {:>10.3}",
            "avg_block_bytes",
            self.stats.sum_block_bytes as f64 / self.len as f64
        );

        println!("{:>20}: {:>10}", "max_lcp", self.stats.max_lcp);
        println!(
            "{:>20}: {:>10.3}",
            "avg_lcp",
            self.stats.sum_lcp as f64 / self.len as f64
        );

        println!("{:>20}: {:>10}", "max_str_len", self.stats.max_str_len);
        println!(
            "{:>20}: {:>10.3}",
            "avg_str_len",
            self.stats.sum_str_len as f64 / self.len as f64
        );

        let ptr_size: usize = self.pointers.len() * core::mem::size_of::<usize>();

        fn human(key: &str, x: isize) {
            const UOM: &[&str] = &["B", "KB", "MB", "GB", "TB"];
            let sign = x.signum();
            let mut y = x.abs() as f64;
            let mut uom_idx = 0;
            while y > 1000.0 {
                uom_idx += 1;
                y /= 1000.0;
            }
            println!(
                "{:>20}:{:>10.3}{}{:>20} ",
                key,
                sign as f64 * y,
                UOM[uom_idx],
                x
            );
        }

        let total_size = ptr_size + self.data.len() + core::mem::size_of::<Self>();
        human("data_bytes", self.data.len() as isize);
        human("codes_bytes", self.stats.code_bytes as isize);
        human("suffixes_bytes", self.stats.suffixes_bytes as isize);
        human("ptrs_bytes", ptr_size as isize);
        human("uncompressed_size", self.stats.sum_str_len as isize);
        human("total_size", total_size as isize);

        human(
            "optimal_size",
            self.data.len() as isize - self.stats.redundancy,
        );
        human("redundancy", self.stats.redundancy);
        let overhead = self.stats.redundancy + ptr_size as isize;
        println!(
            "overhead_ratio: {:>10}",
            overhead as f64 / (overhead + self.data.len() as isize) as f64
        );
        println!(
            "no_overhead_compression_ratio: {:.3}",
            (self.data.len() as isize - self.stats.redundancy) as f64
                / self.stats.sum_str_len as f64
        );

        println!(
            "compression_ratio: {:.3}",
            (ptr_size + self.data.len()) as f64 / self.stats.sum_str_len as f64
        );
    }

    fn push_impl(&mut self, string: impl AsRef<[u8]>) {
        let string = string.as_ref();

        // update stats
        self.stats.max_str_len = self.stats.max_str_len.max(string.len());
        self.stats.sum_str_len += string.len();

        let (lcp, order) = longest_common_prefix(&self.last_str, string);

        if SORTED && order == core::cmp::Ordering::Greater {
            panic!(
                "Strings must be sorted in ascending order when building a sorted FrontCodedList. Got '{:?}' after '{}'",
                string,
                String::from_utf8_lossy(&self.last_str)
            );
        }

        // compute how long is this string without the lcp
        let suffix_len = string.len() - lcp;
        let length_before = self.data.len();

        // at every multiple of ratio we just encode the string as is
        let to_encode = if self.len % self.ratio == 0 {
            // compute the size in bytes of the previous block
            let last_ptr = self.pointers.last().copied().unwrap_or(0);
            let block_bytes = self.data.len() - last_ptr;
            // update stats
            self.stats.max_block_bytes = self.stats.max_block_bytes.max(block_bytes);
            self.stats.sum_block_bytes += block_bytes;
            // save a pointer to the start of the string
            self.pointers.push(self.written_bytes);

            let prev_len = self.data.len();
            // encode the length of the string
            encode_int(string.len(), &mut self.data);
            // update stats
            self.stats.code_bytes += self.data.len() - prev_len;

            // compute the redundancy
            if self.len != 0 {
                self.stats.redundancy += lcp as isize;
                self.stats.redundancy += encode_int_len(string.len()) as isize;
                self.stats.redundancy -= encode_int_len(lcp) as isize;
                self.stats.redundancy -= encode_int_len(suffix_len) as isize;
            }
            // just encode the whole string
            string
        } else {
            // update the stats
            self.stats.max_lcp = self.stats.max_lcp.max(lcp);
            self.stats.sum_lcp += lcp;
            // encode the len of the bytes in data and the lcp
            let prev_len = self.data.len();
            encode_int(suffix_len, &mut self.data);
            encode_int(lcp, &mut self.data);
            // update stats
            self.stats.code_bytes += self.data.len() - prev_len;
            // return the delta suffix
            &string[lcp..]
        };
        // Write the data to the buffer
        self.data.extend_from_slice(to_encode);
        self.written_bytes += self.data.len() - length_before;
        self.stats.suffixes_bytes += to_encode.len();

        // put the string as last_str for the next iteration
        self.last_str.truncate(lcp);
        self.last_str.extend_from_slice(&string[lcp..]);
        self.len += 1;
    }
}

impl<const SORTED: bool> FrontCodedListBuilder<str, SORTED> {
    /// Builds a front-coded list of strings.
    pub fn build(self) -> FrontCodedListStr<SORTED> {
        FrontCodedList {
            data: self.data.into(),
            pointers: self.pointers.into(),
            len: self.len,
            ratio: self.ratio,
            _marker_i: PhantomData,
            _marker_o: PhantomData,
        }
    }
}

impl<const SORTED: bool> FrontCodedListBuilder<[u8], SORTED> {
    /// Builds a front-coded list of slices of bytes (`[u8]`).
    pub fn build(self) -> FrontCodedListSliceU8<SORTED> {
        FrontCodedList {
            data: self.data.into(),
            pointers: self.pointers.into(),
            len: self.len,
            ratio: self.ratio,
            _marker_i: PhantomData,
            _marker_o: PhantomData,
        }
    }
}

impl<I: ?Sized + AsRef<[u8]>, const SORTED: bool> FrontCodedListBuilder<I, SORTED> {
    /// Appends a string to the end of the list.
    ///
    /// # Panics
    ///
    /// Panics if `SORTED` is `true` and the string is not greater than or equal
    /// to the last string added.
    pub fn push(&mut self, string: impl Borrow<I>) {
        self.push_impl(string.borrow().as_ref());
    }

    /// Appends all the elements from a [`Lender`] to the end of the list.
    ///
    /// We prefer to implement extension via a [`Lender`] instead of an
    /// [`Iterator`] to avoid the need to allocate a new string for every string
    /// in the list. This is particularly useful when building large lists from
    /// files using, for example, a
    /// [`FallibleRewindableLender`](crate::utils::FallibleRewindableLender).
    ///
    /// # Panics
    ///
    /// Panics if `SORTED` is `true` and any string is not greater than or equal
    /// to the previous string.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sux::dict::FrontCodedListBuilder;
    ///
    /// // For sorted lists
    /// let mut fclb = FrontCodedListBuilder::<str, true>::new(4);
    /// fclb.push("aa");
    /// fclb.push("aab");
    /// fclb.push("abc");
    /// fclb.push("abdd");
    /// let fcl = fclb.build();
    /// assert_eq!(fcl.len(), 4);
    ///
    /// // For unsorted lists
    /// let mut fclb = FrontCodedListBuilder::<str, false>::new(4);
    /// for word in ["foo", "bar", "baz", "qux"] {
    ///     fclb.push(word);
    /// }
    /// let fcl = fclb.build();
    /// assert_eq!(fcl.len(), 4);
    /// ```
    pub fn extend<L: IntoLender>(&mut self, into_lender: L)
    where
        for<'lend> lender::Lend<'lend, <L as IntoLender>::Lender>: AsRef<I>,
    {
        for_!(elem in into_lender {
            self.push(elem.as_ref());
        });
    }
}

#[inline(always)]
/// Computes the longest common prefix between two slices of bytes.
fn longest_common_prefix(a: &[u8], b: &[u8]) -> (usize, core::cmp::Ordering) {
    let min_len = a.len().min(b.len());
    // normal lcp computation
    let mut i = 0;
    // we purposely don't do early stopping so it can be vectorized
    while i < min_len && a[i] == b[i] {
        i += 1;
    }
    // TODO!: try to optimize with vpcmpeqb pextrb and leading count ones
    if i < min_len {
        (i, a[i].cmp(&b[i]))
    } else {
        (i, a.len().cmp(&b.len()))
    }
}

/// Computes the length in bytes of value encoded as VByte
#[inline(always)]
fn encode_int_len(mut value: usize) -> usize {
    let mut len = 1;
    loop {
        value >>= 7;
        if value == 0 {
            return len;
        }
        value -= 1;
        len += 1;
    }
}

/// VByte encode an integer
#[inline(always)]
fn encode_int(mut value: usize, data: &mut Vec<u8>) {
    let mut buf = [0u8; 10];
    let mut pos = buf.len() - 1;
    buf[pos] = (value & 0x7F) as u8;
    value >>= 7;
    while value != 0 {
        value -= 1;
        pos -= 1;
        buf[pos] = 0x80 | (value & 0x7F) as u8;
        value >>= 7;
    }
    for &byte in &buf[pos..] {
        data.push(byte);
    }
}

#[inline(always)]
fn decode_int(mut data: &[u8]) -> (usize, &[u8]) {
    let mut byte = data[0];
    data = &data[1..];
    let mut value = (byte & 0x7F) as usize;
    while (byte >> 7) != 0 {
        value += 1;
        byte = data[0];
        data = &data[1..];
        value = (value << 7) | (byte & 0x7F) as usize;
    }
    (value, data)
}

// Structures for on-the-fly serialization with ε-serde
#[cfg(feature = "epserde")]
mod epserde_impl {
    use super::{FrontCodedList, FrontCodedListBuilder};
    use crate::utils::FallibleRewindableLender;
    #[cfg(feature = "epserde")]
    use epserde::{prelude::SerIter, ser::Serialize};
    use lender::FallibleLending;
    use std::{borrow::Borrow, cell::RefCell, marker::PhantomData};

    /// Serializes to a stream a front-coded list directly from a lender of `AsRef<[u8]>`.
    fn serialize_impl<
        T: ?Sized + Borrow<I>,
        I: PartialEq<O> + PartialEq + ?Sized + AsRef<[u8]>,
        O: PartialEq<I> + PartialEq,
        L: FallibleRewindableLender<
                RewindError: std::error::Error + Send + Sync + 'static,
                Error: std::error::Error + Send + Sync + 'static,
            > + for<'lend> FallibleLending<'lend, Lend = &'lend T>,
        const SORTED: bool,
    >(
        ratio: usize,
        mut lender: L,
        mut writer: impl std::io::Write,
    ) -> anyhow::Result<usize>
    where
        for<'a> super::FrontCodedList<
            I,
            O,
            SerIter<u8, BytesIter<'a, T, I, L, SORTED>>,
            SerIter<usize, PointersIter<'a, I, SORTED>>,
            SORTED,
        >: Serialize,
    {
        use log::info;

        let mut len = 0;
        let mut byte_len = 0;
        // First pass: count the number of strings and the byte length
        let mut builder = FrontCodedListBuilder::<I, SORTED>::new(ratio);
        info!("Counting strings...");
        while let Some(s) = lender.next()? {
            builder.push(s.borrow());
            len += 1;
            byte_len += builder.data.len();
            builder.data.truncate(0);
        }
        info!("Counted {} strings, compressed in {} bytes", len, byte_len);

        lender = lender.rewind()?;

        // We now use a feature of ε-serde that allows us to serialize iterators.
        // Since we have two interdependent iterators (the bytes and the pointers),
        // we give to each iterator a reference to a RefCell containing a
        // builder.
        let builder = RefCell::new(FrontCodedListBuilder::<I, SORTED>::new(ratio));

        let front_coded_list = FrontCodedList::<I, O, _, _, SORTED> {
            ratio,
            len,
            data: SerIter::new(BytesIter::<T, I, L, SORTED> {
                builder: &builder,
                lender,
                byte_len,
                pos: 0,
                _marker_i: PhantomData,
            }),
            pointers: SerIter::new(PointersIter::<I, SORTED> {
                builder: &builder,
                pos: 0,
            }),
            _marker_i: PhantomData,
            _marker_o: PhantomData,
        };

        info!("Serializing...");
        // SAFETY: There is no padding.
        let written = unsafe { front_coded_list.serialize(&mut writer)? };
        info!("Completed.");
        Ok(written)
    }

    /// Serializes strings to a stream a front-coded list directly from a lender of `AsRef<str>`.
    #[cfg(feature = "epserde")]
    pub fn serialize_str<
        T: ?Sized + Borrow<str>,
        L: FallibleRewindableLender<
                RewindError: std::error::Error + Send + Sync + 'static,
                Error: std::error::Error + Send + Sync + 'static,
            > + for<'lend> FallibleLending<'lend, Lend = &'lend T>,
        const SORTED: bool,
    >(
        ratio: usize,
        lender: L,
        writer: impl std::io::Write,
    ) -> anyhow::Result<usize> {
        serialize_impl::<T, str, String, L, SORTED>(ratio, lender, writer)
    }

    /// Serializes strings to a stream a front-coded list directly from a lender of `AsRef<[u8]>`.
    #[cfg(feature = "epserde")]
    pub fn serialize_slice_u8<
        T: ?Sized + Borrow<[u8]>,
        L: FallibleRewindableLender<
                RewindError: std::error::Error + Send + Sync + 'static,
                Error: std::error::Error + Send + Sync + 'static,
            > + for<'lend> FallibleLending<'lend, Lend = &'lend T>,
        const SORTED: bool,
    >(
        ratio: usize,
        lender: L,
        writer: impl std::io::Write,
    ) -> anyhow::Result<usize> {
        serialize_impl::<T, [u8], Vec<u8>, L, SORTED>(ratio, lender, writer)
    }

    /// Stores into a file a front-coded list of strings built directly from a lender of
    /// `AsRef<str]>`.
    #[cfg(feature = "epserde")]
    pub fn store_str<
        T: ?Sized + Borrow<str>,
        L: FallibleRewindableLender<
                RewindError: std::error::Error + Send + Sync + 'static,
                Error: std::error::Error + Send + Sync + 'static,
            > + for<'lend> FallibleLending<'lend, Lend = &'lend T>,
        const SORTED: bool,
    >(
        ratio: usize,
        lender: L,
        filename: impl AsRef<std::path::Path>,
    ) -> anyhow::Result<usize> {
        let dst_file =
            std::fs::File::create(filename.as_ref()).expect("Cannot create destination file");
        let mut buf_writer = std::io::BufWriter::new(dst_file);
        serialize_impl::<T, str, String, L, SORTED>(ratio, lender, &mut buf_writer)
    }

    /// Stores into a file a front-coded list of strings built directly from a lender of
    /// `AsRef<[u8]>`.
    #[cfg(feature = "epserde")]
    pub fn store_slice_u8<
        T: ?Sized + Borrow<[u8]>,
        L: FallibleRewindableLender<
                RewindError: std::error::Error + Send + Sync + 'static,
                Error: std::error::Error + Send + Sync + 'static,
            > + for<'lend> FallibleLending<'lend, Lend = &'lend T>,
        const SORTED: bool,
    >(
        ratio: usize,
        lender: L,
        filename: impl AsRef<std::path::Path>,
    ) -> anyhow::Result<usize> {
        let dst_file =
            std::fs::File::create(filename.as_ref()).expect("Cannot create destination file");
        let mut buf_writer = std::io::BufWriter::new(dst_file);
        serialize_impl::<T, [u8], Vec<u8>, L, SORTED>(ratio, lender, &mut buf_writer)
    }

    /// An iterator that will be wrapped in a [`SerIter`] to serialize directly the
    /// bytes of the front-coded list.
    ///
    /// We slightly abuse the builder by deleting its data it appends for each
    /// string after reading it. In this way we never allocate more memory than that
    /// needed for a string.
    struct BytesIter<
        'a,
        T: ?Sized + Borrow<I>,
        I: ?Sized + AsRef<[u8]>,
        L: FallibleRewindableLender<
                RewindError: std::error::Error + Send + Sync + 'static,
                Error: std::error::Error + Send + Sync + 'static,
            > + for<'lend> FallibleLending<'lend, Lend = &'lend T>,
        const SORTED: bool,
    > {
        builder: &'a RefCell<FrontCodedListBuilder<I, SORTED>>,
        lender: L,
        byte_len: usize,
        pos: usize,
        _marker_i: PhantomData<I>,
    }

    impl<
        'a,
        T: ?Sized + Borrow<I>,
        I: ?Sized + AsRef<[u8]>,
        L: FallibleRewindableLender<
                RewindError: std::error::Error + Send + Sync + 'static,
                Error: std::error::Error + Send + Sync + 'static,
            > + for<'lend> FallibleLending<'lend, Lend = &'lend T>,
        const SORTED: bool,
    > Iterator for BytesIter<'a, T, I, L, SORTED>
    {
        type Item = u8;

        #[inline(always)]
        fn next(&mut self) -> Option<Self::Item> {
            let mut builder = self.builder.borrow_mut();
            if self.pos < builder.data.len() {
                // There's still data in the builder--just return the next byte
                let byte = builder.data[self.pos];
                self.pos += 1;
                self.byte_len -= 1;
                Some(byte)
            } else {
                match self.lender.next() {
                    Ok(None) => None,
                    Ok(Some(s)) => {
                        // Empty the builder data and refill it
                        builder.data.truncate(0);
                        builder.push(s.borrow());
                        let byte = builder.data[0];
                        self.pos = 1;
                        self.byte_len -= 1;
                        Some(byte)
                    }

                    Err(_e) => {
                        panic!("Error while serializing FrontCodedList")
                    }
                }
            }
        }
    }

    impl<
        'a,
        T: ?Sized + Borrow<I>,
        I: ?Sized + AsRef<[u8]>,
        L: FallibleRewindableLender<
                RewindError: std::error::Error + Send + Sync + 'static,
                Error: std::error::Error + Send + Sync + 'static,
            > + for<'lend> FallibleLending<'lend, Lend = &'lend T>,
        const SORTED: bool,
    > ExactSizeIterator for BytesIter<'a, T, I, L, SORTED>
    {
        fn len(&self) -> usize {
            self.byte_len
        }
    }

    /// An iterator that will be wrapped in a [`SerIter`] to serialize directly the
    /// pointers of the front-coded list.
    ///
    /// We just need to read the pointers from the builder. The field `pos` keeps
    /// track of the position of the next pointer to write.
    struct PointersIter<'a, I: ?Sized + AsRef<[u8]>, const SORTED: bool> {
        builder: &'a RefCell<FrontCodedListBuilder<I, SORTED>>,
        pos: usize,
    }

    impl<'a, I: ?Sized + AsRef<[u8]>, const SORTED: bool> Iterator for PointersIter<'a, I, SORTED> {
        type Item = usize;

        fn next(&mut self) -> Option<Self::Item> {
            let builder = self.builder.borrow();
            if self.pos == builder.pointers.len() {
                None
            } else {
                let ptr = builder.pointers[self.pos];
                self.pos += 1;
                Some(ptr)
            }
        }
    }

    impl<'a, I: ?Sized + AsRef<[u8]>, const SORTED: bool> ExactSizeIterator
        for PointersIter<'a, I, SORTED>
    {
        fn len(&self) -> usize {
            let builder = self.builder.borrow();
            builder.pointers.len() - self.pos
        }
    }
}

#[cfg(feature = "epserde")]
pub use epserde_impl::{serialize_slice_u8, serialize_str, store_slice_u8, store_str};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memcmp() {
        assert_eq!(memcmp(b"abcd", b"abcd"), core::cmp::Ordering::Equal);
        assert_eq!(memcmp(b"abcd", b"abbd"), core::cmp::Ordering::Greater);
        assert_eq!(memcmp(b"abcd", b"abdd"), core::cmp::Ordering::Less);

        assert_eq!(memcmp(b"a", b"b\0"), core::cmp::Ordering::Less);
        assert_eq!(memcmp(b"b", b"a\0"), core::cmp::Ordering::Greater);
        assert_eq!(memcmp(b"abc", b"abc\0"), core::cmp::Ordering::Less);
        assert_eq!(memcmp(b"abc", b"ab\0"), core::cmp::Ordering::Greater);
        assert_eq!(memcmp(b"ab\0", b"abc"), core::cmp::Ordering::Less);
    }

    const UPPER_BOUND_1: usize = 128;
    const UPPER_BOUND_2: usize = 128_usize.pow(2) + UPPER_BOUND_1;
    const UPPER_BOUND_3: usize = 128_usize.pow(3) + UPPER_BOUND_2;
    const UPPER_BOUND_4: usize = 128_usize.pow(4) + UPPER_BOUND_3;
    const UPPER_BOUND_5: usize = 128_usize.pow(5) + UPPER_BOUND_4;
    const UPPER_BOUND_6: usize = 128_usize.pow(6) + UPPER_BOUND_5;
    const UPPER_BOUND_7: usize = 128_usize.pow(7) + UPPER_BOUND_6;
    const UPPER_BOUND_8: usize = 128_usize.pow(8) + UPPER_BOUND_7;

    #[test]
    fn test_encode_decode_int() {
        let values = [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            UPPER_BOUND_1 - 1,
            UPPER_BOUND_1,
            UPPER_BOUND_1 + 1,
            UPPER_BOUND_2 - 1,
            UPPER_BOUND_2,
            UPPER_BOUND_2 + 1,
            UPPER_BOUND_3 - 1,
            UPPER_BOUND_3,
            UPPER_BOUND_3 + 1,
            UPPER_BOUND_4 - 1,
            UPPER_BOUND_4,
            UPPER_BOUND_4 + 1,
            UPPER_BOUND_5 - 1,
            UPPER_BOUND_5,
            UPPER_BOUND_5 + 1,
            UPPER_BOUND_6 - 1,
            UPPER_BOUND_6,
            UPPER_BOUND_6 + 1,
            UPPER_BOUND_7 - 1,
            UPPER_BOUND_7,
            UPPER_BOUND_7 + 1,
            UPPER_BOUND_8 - 1,
            UPPER_BOUND_8,
            UPPER_BOUND_8 + 1,
        ];
        let mut buffer = Vec::with_capacity(128);

        for i in &values {
            encode_int(*i, &mut buffer);
        }

        let mut data = &buffer[..];
        for i in &values {
            let (j, tmp) = decode_int(data);
            assert_eq!(data.len() - tmp.len(), encode_int_len(*i));
            data = tmp;
            assert_eq!(*i, j);
        }
    }

    #[test]
    fn test_longest_common_prefix() {
        let str1 = b"absolutely";
        let str2 = b"absorption";
        assert_eq!(
            longest_common_prefix(str1, str2),
            (4, core::cmp::Ordering::Less),
        );
        assert_eq!(
            longest_common_prefix(str1, str1),
            (str1.len(), core::cmp::Ordering::Equal)
        );
        assert_eq!(
            longest_common_prefix(str2, str2),
            (str2.len(), core::cmp::Ordering::Equal)
        );
    }
}
