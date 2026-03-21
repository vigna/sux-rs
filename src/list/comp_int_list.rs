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
//! cumulative bit positions in an [Elias–Fano
//! sequence](crate::dict::elias_fano::EfSeq), enabling efficient random access.
//!
//! The delimiter structure can be [replaced](CompIntList::map_delimiters) with
//! any structure implementing [`IntoIteratorFrom`] with the returned iterator
//! implementing `UncheckedIterator<Item = u64>`, as long as it returns the same
//! cumulative bit positions.
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

use mem_dbg::*;
use value_traits::slices::SliceByValue;

use crate::dict::EliasFanoBuilder;
use crate::dict::elias_fano::EfSeq;
use crate::traits::Word;
use crate::traits::iter::{IntoIteratorFrom, UncheckedIterator};
use crate::utils::PrimitiveUnsignedExt;

/// A compact list of integers not less than a given lower bound.
///
/// Each value *x* is stored as *x* − `min` + 1 (a strictly positive integer)
/// by concatenating binary representations with the most significant bit
/// removed. The boundaries between values are recorded in a
/// [`SliceByValue<Value = u64>`] (by default an [Elias–Fano
/// sequence](EfSeq)), enabling efficient random access.
///
/// After construction, the delimiter structure can be replaced using
/// [`map_delimiters`](CompIntList::map_delimiters).
///
/// # Type Parameters
///
/// - `V`: The value type. Must be a [`Word`] type. Defaults to `u64`.
/// - `D`: The delimiter structure. Must implement `SliceByValue<Value = u64>`.
///   Defaults to [`EfSeq`].
#[derive(Debug, Clone, MemDbg, MemSize)]
pub struct CompIntList<V: Word = usize, D: SliceByValue<Value = V> = EfSeq<V>> {
    /// Number of stored values.
    n: usize,
    /// Lower bound on the values.
    min: V,
    /// Structure storing `n + 1` cumulative bit-position delimiters.
    delimiters: D,
    /// Concatenated binary representations (MSB removed), backed by
    /// `V`-sized words.
    data: Box<[V]>,
}

impl<V: Word> CompIntList<V> {
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
    pub fn new<I: ?Sized>(min: V, values: &I) -> Self
    where
        for<'a> &'a I: IntoIterator<Item = &'a V>,
    {
        // First pass: count elements and total bits
        let mut n = 0usize;
        let mut total_bits = 0u64;
        for &v in values {
            assert!(v >= min, "CompIntList: value must be >= the lower bound");
            let offset = v - min + V::ONE;
            total_bits += (offset.bit_len() - 1) as u64;
            n += 1;
        }

        // Second pass: build delimiters and pack data
        let mut efb = EliasFanoBuilder::new(n + 1, total_bits);
        let mut pos = 0u64;
        // SAFETY: pos = 0 ≤ total_bits and is the first push
        unsafe { efb.push_unchecked(0u64) };

        // Allocate data buffer (at least one word for safe two-word reads)
        let n_words = (total_bits as usize).div_ceil(V::BITS as usize) + 1;
        let mut data = vec![V::ZERO; n_words];
        let mut bit_pos = 0usize;

        for &v in values {
            let offset = v - min + V::ONE;
            let width = (offset.bit_len() - 1) as usize;
            if width > 0 {
                let bits = offset ^ (V::ONE << width);
                Self::write_bits(&mut data, bit_pos, bits, width);
            }
            bit_pos += width;
            pos += width as u64;
            // SAFETY: pos is non-decreasing and ≤ total_bits
            unsafe { efb.push_unchecked(pos) };
        }
        let delimiters = efb.build_with_seq();

        CompIntList {
            n,
            min,
            delimiters,
            data: data.into_boxed_slice(),
        }
    }

    /// Writes `width` low bits of `value` into `data` starting at bit
    /// position `start`. The value must fit in `width` bits and `width`
    /// must be less than `V::BITS`.
    fn write_bits(data: &mut [V], start: usize, value: V, width: usize) {
        let v_bits = V::BITS as usize;
        let word_idx = start / v_bits;
        let bit_idx = start % v_bits;

        data[word_idx] |= value << bit_idx;
        if bit_idx + width > v_bits {
            data[word_idx + 1] |= value >> (v_bits - bit_idx);
        }
    }
}

impl<V: Word, D: SliceByValue<Value = u64>> CompIntList<V, D> {
    /// Replaces the delimiter structure.
    ///
    /// # Safety
    ///
    /// This method is unsafe because it is not possible to guarantee that the
    /// new delimiters return the same values as the old ones.
    pub unsafe fn map_delimiters<F, D2>(self, func: F) -> CompIntList<V, D2>
    where
        F: FnOnce(D) -> D2,
        D2: SliceByValue<Value = u64>,
    {
        CompIntList {
            n: self.n,
            min: self.min,
            delimiters: func(self.delimiters),
            data: self.data,
        }
    }

    /// Returns the underlying delimiter structure.
    pub fn into_inner(self) -> D {
        self.delimiters
    }

    /// Reads `width` bits from `data` starting at bit position `start`.
    ///
    /// # Safety
    ///
    /// - `start + width` must not exceed the total number of bits in `data`.
    /// - `width` must be less than `V::BITS`.
    #[inline(always)]
    unsafe fn read_bits(data: &[V], start: usize, width: usize) -> V {
        let v_bits = V::BITS as usize;
        let word_idx = start / v_bits;
        let bit_idx = start % v_bits;
        let mask = (V::ONE << width) - V::ONE;

        unsafe {
            if bit_idx + width <= v_bits {
                (*data.get_unchecked(word_idx) >> bit_idx) & mask
            } else {
                ((*data.get_unchecked(word_idx) >> bit_idx)
                    | (*data.get_unchecked(word_idx + 1) << (v_bits - bit_idx)))
                    & mask
            }
        }
    }
}

impl<V: Word, D: SliceByValue<Value = u64>> SliceByValue for CompIntList<V, D>
where
    for<'a> &'a D: IntoIteratorFrom,
    for<'a> <&'a D as IntoIteratorFrom>::IntoIterFrom: UncheckedIterator<Item = u64>,
{
    type Value = V;

    #[inline(always)]
    fn len(&self) -> usize {
        self.n
    }

    #[inline(always)]
    unsafe fn get_value_unchecked(&self, index: usize) -> V {
        let mut iter = (&self.delimiters).into_iter_from(index);
        let start = unsafe { iter.next_unchecked() } as usize;
        let end = unsafe { iter.next_unchecked() } as usize;
        let width = end - start;

        let bits = unsafe { Self::read_bits(&self.data, start, width) };
        let stored = (V::ONE << width) | bits;

        // stored = value - min + 1, so value = (stored - 1) + min
        (stored - V::ONE) + self.min
    }
}
