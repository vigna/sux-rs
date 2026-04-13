/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! A compact list of integers not less than a given lower bound, based on
//! delimiters.
//!
//! Given a lower bound `min` and a list of integers not less than `min`, a
//! [`CompIntList`] stores each value *x* as *x* − `min` + 1, which is strictly
//! positive. These offset values are then concatenated in binary with the most
//! significant bit removed (since it is always 1). For an offset value *y* =
//! *x* − `min` + 1 > 0, the number of stored bits is ⌊log₂ *y*⌋.
//!
//! The boundaries between these variable-length representations are recorded as
//! cumulative bit positions in an [Elias–Fano sequence], enabling efficient
//! random access.
//!
//! [Elias–Fano sequence]: crate::dict::elias_fano::EfSeq
//!
//! The delimiter structure can be [replaced][map_delimiters] with any
//! structure implementing [`IntoIteratorFrom`] with the returned iterator
//! implementing `UncheckedIterator<Item = u64>`, as long as it returns the
//! same cumulative bit positions.
//!
//! The data structure can be [replaced][map_data] with any structure
//! implementing [`BitVecValueOps`].
//!
//! This structure implements [`SliceByValue`] for random access.
//!
//! # Examples
//!
//! ```rust
//! use sux::list::comp_int_list::CompIntList;
//! use value_traits::slices::SliceByValue;
//!
//! let values = vec![1u64, 3, 7, 42, 100];
//! let list = CompIntList::new(1, &values);
//!
//! assert_eq!(list.index_value(0), 1);
//! assert_eq!(list.index_value(1), 3);
//! assert_eq!(list.index_value(2), 7);
//! assert_eq!(list.index_value(3), 42);
//! assert_eq!(list.index_value(4), 100);
//! ```
//!
//! [map_delimiters]: CompIntList::map_delimiters
//! [map_data]: CompIntList::map_data

use crate::bits::bit_vec::{BitVec, BitVecU};
use crate::bits::test_unaligned_any_pos;
use crate::dict::EliasFanoBuilder;
use crate::dict::elias_fano::EfSeq;
use crate::traits::iter::{IntoIteratorFrom, UncheckedIterator};
use crate::traits::{Backend, BitVecValueOps, TryIntoUnaligned, Word};
use crate::utils::PrimitiveUnsignedExt;
use mem_dbg::*;
use value_traits::slices::SliceByValue;

/// A compact list of integers not less than a given lower bound.
///
/// Each value *x* is stored as *x* − `min` + 1 (a strictly positive integer)
/// by concatenating binary representations with the most significant bit
/// removed. The boundaries between values are recorded in a
/// [`SliceByValue<Value = u64>`] (by default an [Elias–Fano sequence]),
/// enabling efficient random access.
///
/// After construction, the delimiter structure can be replaced using
/// [`map_delimiters`], and the data structure using [`map_data`].
///
/// This structure implements the [`TryIntoUnaligned`]
/// trait, allowing it to be converted into (usually faster) structures using
/// unaligned access.
///
/// # Type Parameters
///
/// - `B`: The data backend. Must implement [`Backend`] and
///   [`BitVecValueOps<B::Word>`]. Defaults to
///   [`BitVec<Box<[usize]>>`].
/// - `D`: The delimiter structure. Must implement `SliceByValue<Value = u64>`.
///   Defaults to [`EfSeq<u64>`].
///
/// [Elias–Fano sequence]: EfSeq
/// [`map_delimiters`]: CompIntList::map_delimiters
/// [`map_data`]: CompIntList::map_data
/// [`EfSeq<u64>`]: EfSeq
#[derive(Debug, Clone, MemSize, MemDbg)]
#[cfg_attr(
    feature = "epserde",
    derive(epserde::Epserde),
    epserde(bound(
        deser = "for<'a> <B as epserde::deser::DeserInner>::DeserType<'a>: Backend<Word = B::Word>"
    ))
)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CompIntList<B: Backend = BitVec<Box<[usize]>>, D = EfSeq<u64>> {
    /// Number of stored values.
    ///
    /// Note that this is identical to `delimiters.len() - 1`, but we have no
    /// guarantee that calling `delimiters.len()` is O(1).
    n: usize,
    /// Lower bound on the values.
    min: B::Word,
    /// Structure storing `n + 1` cumulative bit-position delimiters.
    delimiters: D,
    /// Concatenated binary representations (MSB removed).
    data: B,
    /// Whether all stored bit widths satisfy the constraints for unaligned
    /// reads.
    all_widths_unaligned: bool,
}

impl<V: Word> CompIntList<BitVec<Box<[V]>>> {
    /// Creates a new `CompIntList` from a lower bound and a reference to a
    /// collection of values not less than `min`.
    ///
    /// The collection is iterated twice: once to compute statistics (element
    /// count and total bit length), and once to build the delimiter and data
    /// structures.
    ///
    /// # Panics
    ///
    /// Panics if any value is less than `min`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use sux::list::comp_int_list::CompIntList;
    /// # use value_traits::slices::SliceByValue;
    /// let values = vec![1u64, 5, 10];
    /// let list = CompIntList::new(1, &values);
    /// assert_eq!(list.len(), 3);
    /// assert_eq!(list.index_value(1), 5);
    /// ```
    #[must_use]
    pub fn new<I: ?Sized>(min: V, values: &I) -> Self
    where
        for<'a> &'a I: IntoIterator<Item = &'a V>,
    {
        // First pass: count elements and total bits
        let mut n = 0;
        let mut total_bits = 0u64;
        let mut all_widths_unaligned = true;
        for &v in values {
            assert!(v >= min, "CompIntList: value must be >= the lower bound");
            let offset = v - min + V::ONE;
            let width = (offset.bit_len() - 1) as usize;
            total_bits += width as u64;
            if !test_unaligned_any_pos!(V, width) {
                all_widths_unaligned = false;
            }
            n += 1;
        }

        // Second pass: build delimiters and pack data
        let mut efb = EliasFanoBuilder::new(n + 1, total_bits);
        let mut pos = 0;
        // SAFETY: pos = 0 ≤ total_bits and is the first push
        unsafe { efb.push_unchecked(0) };

        let mut data: BitVec<Vec<V>> = BitVec::new(0);
        data.reserve(total_bits as usize);

        for &v in values {
            let offset = v - min + V::ONE;
            let width = (offset.bit_len() - 1) as usize;
            let bits = offset ^ (V::ONE << width);
            data.append_value(bits, width);
            pos += width as u64;
            // SAFETY: pos is non-decreasing and ≤ total_bits
            unsafe { efb.push_unchecked(pos) };
        }
        let delimiters = efb.build_with_seq();
        let data = data.into_padded();

        CompIntList {
            n,
            min,
            delimiters,
            data,
            all_widths_unaligned,
        }
    }
}

impl<B: Backend<Word: Word> + BitVecValueOps<B::Word>, D: SliceByValue<Value = u64>>
    CompIntList<B, D>
{
    /// Replaces the delimiter structure.
    ///
    /// # Safety
    ///
    /// This method is unsafe because it is not possible to guarantee that the
    /// new delimiters return the same values as the old ones.
    pub unsafe fn map_delimiters<F, D2>(self, func: F) -> CompIntList<B, D2>
    where
        F: FnOnce(D) -> D2,
        D2: SliceByValue<Value = u64>,
    {
        CompIntList {
            n: self.n,
            min: self.min,
            delimiters: func(self.delimiters),
            data: self.data,
            all_widths_unaligned: self.all_widths_unaligned,
        }
    }

    /// Replaces the data structure.
    ///
    /// # Safety
    ///
    /// This method is unsafe because it is not possible to guarantee that the
    /// new data returns the same values as the old one.
    pub unsafe fn map_data<F, B2>(self, func: F) -> CompIntList<B2, D>
    where
        F: FnOnce(B) -> B2,
        B2: Backend<Word = B::Word> + BitVecValueOps<B2::Word>,
    {
        CompIntList {
            n: self.n,
            min: self.min,
            delimiters: self.delimiters,
            data: func(self.data),
            all_widths_unaligned: self.all_widths_unaligned,
        }
    }

    /// Returns the underlying delimiter structure.
    pub fn into_inner(self) -> D {
        self.delimiters
    }
}

impl<B: Backend<Word: Word> + BitVecValueOps<B::Word>, D: SliceByValue<Value = u64>> SliceByValue
    for CompIntList<B, D>
where
    for<'a> &'a D: IntoIteratorFrom,
    for<'a> <&'a D as IntoIteratorFrom>::IntoIterFrom: UncheckedIterator<Item = u64>,
{
    type Value = B::Word;

    #[inline(always)]
    fn len(&self) -> usize {
        self.n
    }

    #[inline]
    unsafe fn get_value_unchecked(&self, index: usize) -> B::Word {
        let mut iter = (&self.delimiters).into_iter_from(index);
        let start = unsafe { iter.next_unchecked() } as usize;
        let end = unsafe { iter.next_unchecked() } as usize;
        let width = end - start;

        let bits = unsafe { self.data.get_value_unchecked(start, width) };
        let stored = (B::Word::ONE << width) | bits;

        // stored = value - min + 1, so value = (stored - 1) + min
        (stored - B::Word::ONE) + self.min
    }
}

impl<V: Word, D: TryIntoUnaligned + SliceByValue<Value = u64>> TryIntoUnaligned
    for CompIntList<BitVec<Box<[V]>>, D>
where
    D::Unaligned: SliceByValue<Value = u64>,
{
    type Unaligned = CompIntList<BitVecU<Box<[V]>>, D::Unaligned>;
    /// This method will fail if any stored value has a bit width that does not
    /// satisfy the constraints for unaligned reads.
    fn try_into_unaligned(
        self,
    ) -> Result<Self::Unaligned, crate::traits::UnalignedConversionError> {
        if !self.all_widths_unaligned {
            return Err(crate::traits::UnalignedConversionError(
                "CompIntList contains values whose bit widths do not satisfy the \
                 constraints for unaligned reads"
                    .to_string(),
            ));
        }

        Ok(CompIntList {
            n: self.n,
            min: self.min,
            delimiters: self.delimiters.try_into_unaligned()?,
            data: self.data.try_into_unaligned()?,
            all_widths_unaligned: true,
        })
    }
}

impl<V: Word, D, D2: SliceByValue<Value = u64>> From<CompIntList<BitVecU<Box<[V]>>, D>>
    for CompIntList<BitVec<Box<[V]>>, D2>
where
    D: Into<D2>,
{
    fn from(c: CompIntList<BitVecU<Box<[V]>>, D>) -> Self {
        CompIntList {
            n: c.n,
            min: c.min,
            delimiters: c.delimiters.into(),
            data: c.data.into(),
            all_widths_unaligned: c.all_widths_unaligned,
        }
    }
}
