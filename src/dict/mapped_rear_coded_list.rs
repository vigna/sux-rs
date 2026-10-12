/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Tommaso Fontana
 * SPDX-FileCopyrightText: 2025 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

//! A [`RearCodedList`] indexed through a mapping.
//!
//! The typical use case of this structure is to compress and access efficiently
//! list of strings that are not in sorted order. You build a sorted
//! [`RearCodedList`], and then you [wrap the instance together with a
//! mapping] that maps the original order
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
//! These standard types store the mapping in a bit-packed [`BitFieldVec`] backed
//! by `Box<[usize]>`. However, it is possible to use any type implementing the
//! [`SliceByValue`] trait from the [`value_traits`] crate.
//!
//! Besides the standard access by means of the [`IndexedSeq`] trait, this
//! structure also implements the `get_in_place` method, which makes it possible
//! to write the element directly into a user-provided buffer (a string or a
//! vector of bytes), avoiding allocations. [`MappedRearCodedListStr`] has an
//! additional [`get_bytes_in_place`] method that writes the bytes of the
//! string into a user-provided `Vec<u8>`.
//!
//! Mapped rear-coded lists can be iterated upon using either an
//! [`Iterator`] or a [`Lender`]. In the first case there will be an
//! allocation at each iteration, whereas in the second case a single buffer
//! will be reused. You can also [iterate from a given position]. The
//! iteration will not be as fast as in the non-mapped case, however, as it
//! is not possible to build the returned strings incrementally.
//!
//! Note that, contrarily to [`RearCodedList`], this structure does not provide
//! implementations for the [`IndexedDict`] trait,
//! independently of whether the underlying [`RearCodedList`] is sorted or not.
//!
//! Finally, the `mrcl` command-line tool (which requires the `cli` feature)
//! can be used to create a serialized [`MappedRearCodedListStr`] starting from
//! a serialized [`RearCodedListStr`] and a file containing a mapping (see
//! [below](#the-mapping) for its direction).
//!
//! # The Mapping
//!
//! The mapping goes from positions in the mapped list to positions in the
//! underlying rear-coded list: the element of index *i* of a mapped rear-coded
//! list is the element of index `map[i]` of the underlying rear-coded list. In
//! the typical use case, `map[i]` is thus the rank in sorted order of the
//! element of index *i* of the original, unsorted list.
//!
//! Sorting the indices of the original list by the elements they refer to
//! yields the inverse permutation: here we use it to push the elements in
//! sorted order and to fill the mapping.
//!
//! ```
//! use sux::bits::BitFieldVec;
//! use sux::dict::{MappedRearCodedListStr, RearCodedListBuilder};
//! use sux::traits::IndexedSeq;
//! use value_traits::slices::SliceByValueMut;
//!
//! let words = ["pear", "apple", "peach", "apricot", "banana"];
//! // perm[r] is the index in words of the element of rank r
//! let mut perm: Vec<usize> = (0..words.len()).collect();
//! perm.sort_unstable_by_key(|&i| words[i]);
//!
//! let mut rclb = RearCodedListBuilder::<str, true>::new(4);
//! let bit_width = words.len().ilog2() as usize + 1;
//! let mut map = BitFieldVec::<Vec<usize>>::new(bit_width, words.len());
//! for (rank, &orig) in perm.iter().enumerate() {
//!     rclb.push(words[orig]);
//!     map.set_value(orig, rank);
//! }
//!
//! let mrcl = MappedRearCodedListStr::from_parts(rclb.build(), map.into());
//! for (i, word) in words.iter().enumerate() {
//!     assert_eq!(mrcl.get(i), *word);
//! }
//! ```
//!
//! The `mrcl` command-line tool reads the mapping from a file containing one
//! value per line, in the order above (i.e., `map[0]`, `map[1]`, …). If the file
//! contains instead the inverse permutation (in the example, `perm[0]`,
//! `perm[1]`, …, that is, for each element of the rear-coded list, its
//! position in the mapped list), use the `--invert` option.
//!
//! # Serialization and the `SORTED` Parameter
//!
//! As in the case of [`RearCodedList`], ε-serde includes the const parameter
//! `SORTED` in the type hash, so a list must be deserialized with the same
//! value of `SORTED` it was serialized with. For example, `mrcl --unsorted`
//! reads a [`RearCodedListStr<false>`] (e.g., built by `rcl --unsorted`) and
//! writes a [`MappedRearCodedListStr<false>`], which you can load with
//! `<MappedRearCodedListStr<false>>::load_full` (or map into memory with
//! `<MappedRearCodedListStr<false>>::mmap`, which requires the `mmap`
//! feature), but not as a [`MappedRearCodedListStr`]:
//!
//! ```
//! # #[cfg(feature = "epserde")] {
//! use epserde::{deser::Deserialize, ser::Serialize};
//! use sux::bits::BitFieldVec;
//! use sux::dict::{MappedRearCodedListStr, RearCodedListBuilder};
//! use sux::traits::IndexedSeq;
//!
//! let mut rclb = RearCodedListBuilder::<str, false>::new(4);
//! rclb.push("foo");
//! rclb.push("bar");
//! let mut map = BitFieldVec::<Vec<usize>>::new(1, 0);
//! map.push(1);
//! map.push(0);
//! let mrcl = MappedRearCodedListStr::<false>::from_parts(rclb.build(), map.into());
//!
//! let dir = tempfile::tempdir()?;
//! let path = dir.path().join("words.mrcl");
//! unsafe { mrcl.store(&path)? };
//! let mrcl = unsafe { <MappedRearCodedListStr<false>>::load_full(&path)? };
//! assert_eq!(mrcl.get(0), "bar");
//! // The default alias has SORTED = true, so the type check fails
//! assert!(unsafe { <MappedRearCodedListStr>::load_full(&path) }.is_err());
//! # }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Examples
//!
//! Here we build a sorted rear-coded list, and then we wrap it in a
//! [`MappedRearCodedListStr`] together with a permutation:
//!
//! ```
//! use sux::bits::BitFieldVec;
//! use sux::dict::{MappedRearCodedListStr, RearCodedListBuilder};
//! use sux::traits::IndexedSeq;
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
//! let mut map = BitFieldVec::<Vec<usize>>::new(4, 0);
//! for &v in &[5, 4, 2, 0, 1, 3] { map.push(v); } // permutation
//! let mrcl = MappedRearCodedListStr::from_parts(rcl, map.into());
//! assert_eq!(mrcl.get(0), "abdf");
//! assert_eq!(mrcl.get(1), "abde\0f");
//! assert_eq!(mrcl.get(2), "abc");
//! assert_eq!(mrcl.get(3), "aa");
//! assert_eq!(mrcl.get(4), "aab");
//! assert_eq!(mrcl.get(5), "abdd");
//! ```
//!
//! [`IndexedDict`]: crate::traits::IndexedDict
//! [`RearCodedListStr`]: crate::dict::RearCodedListStr
//! [`RearCodedListStr<false>`]: crate::dict::RearCodedListStr
//! [`get_bytes_in_place`]: MappedRearCodedListStr::get_bytes_in_place
//! [`Iterator`]: MappedRearCodedList::iter
//! [`Lender`]: MappedRearCodedList::lender
//! [iterate from a given position]: MappedRearCodedList::lender_from
//! [wrap the instance together with a mapping]: MappedRearCodedList::from_parts
use crate::bits::BitFieldVec;
use crate::dict::rear_coded_list::{RearCodedList, impl_iteration};
use crate::traits::{IndexedSeq, IntoIteratorFrom, Types};
use lender::FusedLender;
use lender::{ExactSizeLender, IntoLender, Lender, Lending, check_covariance};
use mem_dbg::*;
use value_traits::slices::SliceByValue;

/// Main structure; please use the type aliases [`MappedRearCodedListStr`] and
/// [`MappedRearCodedListSliceU8`].
///
/// Instances are built by combining a [`RearCodedList`] (see
/// [`RearCodedListBuilder`]) and a mapping with [`from_parts`]; see the [module
/// documentation] for the direction of the mapping and for examples.
///
/// [`RearCodedListBuilder`]: crate::dict::RearCodedListBuilder
/// [`from_parts`]: MappedRearCodedList::from_parts
/// [module documentation]: crate::dict::mapped_rear_coded_list
#[derive(Debug, Clone, MemSize, MemDbg)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde), epserde(phantom(I, O)))]
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

/// A mapped rear-coded list of byte sequences, returning `Vec<u8>` on access,
/// with the mapping stored in a [`BitFieldVec`].
///
/// Instances are built from a [`RearCodedListSliceU8<SORTED>`] and a mapping
/// using [`from_parts`]. When deserializing with ε-serde, `SORTED` must match
/// the value used at construction time (see the [module documentation]).
///
/// [`RearCodedListSliceU8<SORTED>`]: crate::dict::RearCodedListSliceU8
/// [`from_parts`]: MappedRearCodedList::from_parts
/// [module documentation]: crate::dict::mapped_rear_coded_list#serialization-and-the-sorted-parameter
pub type MappedRearCodedListSliceU8<const SORTED: bool = true> =
    MappedRearCodedList<[u8], Vec<u8>, Box<[u8]>, Box<[usize]>, BitFieldVec<Box<[usize]>>, SORTED>;

/// A mapped rear-coded list of strings, returning [`String`] on access, with
/// the mapping stored in a [`BitFieldVec`].
///
/// Instances are built from a [`RearCodedListStr<SORTED>`] and a mapping using
/// [`from_parts`]. When deserializing with ε-serde, `SORTED` must match the
/// value used at construction time (see the [module documentation]).
///
/// [`RearCodedListStr<SORTED>`]: crate::dict::RearCodedListStr
/// [`from_parts`]: MappedRearCodedList::from_parts
/// [module documentation]: crate::dict::mapped_rear_coded_list#serialization-and-the-sorted-parameter
pub type MappedRearCodedListStr<const SORTED: bool = true> =
    MappedRearCodedList<str, String, Box<[u8]>, Box<[usize]>, BitFieldVec<Box<[usize]>>, SORTED>;

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
    /// and that all indices are in range. This method is safe, so an
    /// invalid mapping cannot cause memory unsafety, but accessor methods
    /// will panic or return unspecified results.
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

    /// Returns the rear-coded list and the mapping, consuming this structure.
    pub fn into_parts(self) -> (RearCodedList<I, O, D, P, SORTED>, Q) {
        (self.rcl, self.map)
    }

    /// Returns the number of elements.
    ///
    /// This method is equivalent to [`IndexedSeq::len`], but it is provided to
    /// reduce ambiguity in method resolution.
    #[inline]
    pub const fn len(&self) -> usize {
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

impl<
    I: PartialEq<O> + PartialEq + ?Sized,
    O: PartialEq<I> + PartialEq,
    D: AsRef<[u8]>,
    P: AsRef<[usize]>,
    Q: SliceByValue<Value = usize>,
    const SORTED: bool,
> IndexedSeq for MappedRearCodedList<I, O, D, P, Q, SORTED>
where
    RearCodedList<I, O, D, P, SORTED>: for<'a> IndexedSeq<Output<'a> = O>,
{
    #[inline(always)]
    unsafe fn get_unchecked(&self, index: usize) -> Self::Output<'_> {
        let index = self.map.index_value(index);
        unsafe { self.rcl.get_unchecked(index) }
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
        // The checked accessor turns an out-of-range mapped value into a
        // clean bounds panic, as in the str variant.
        self.rcl.get_in_place(index, result);
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
    /// [`str::from_utf8_unchecked`] to avoid the cost of UTF-8 checks. Be aware,
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
    /// [`str::from_utf8_unchecked`] to avoid the cost of UTF-8 checks. Be aware,
    /// however, that using invalid UTF-8 data may lead to undefined behavior.
    #[inline(always)]
    pub fn get_bytes_in_place(&self, index: usize, result: &mut Vec<u8>) {
        let index = self.map.index_value(index);
        // The checked accessor turns an out-of-range mapped value into a
        // clean bounds panic, as in get_bytes.
        self.rcl.get_bytes_in_place(index, result);
    }
}

// Lenders

/// Sequential [`Lender`] over the contents of the list.
#[derive(Debug, Clone, MemSize, MemDbg)]
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
        assert!(
            from <= prcl.len(),
            "Index out of bounds: {} > {}",
            from,
            prcl.len()
        );
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

impl_iteration!(
    MappedRearCodedList,
    prcl,
    D: AsRef<[u8]>,
    P: AsRef<[usize]>,
    Q: SliceByValue<Value = usize>
);
