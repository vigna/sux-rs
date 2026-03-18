/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! A compact list of strictly positive integers based on
//! [Elias–Fano](crate::dict::EliasFano) delimiters.
//!
//! Given a list of strictly positive integers, the [`CompIntList`] stores them by
//! concatenating their binary representations with the most significant bit
//! removed (since it is always 1). The boundaries between these
//! variable-length representations are recorded as cumulative bit positions
//! in an [Elias–Fano sequence](crate::dict::elias_fano::EfSeq), enabling
//! efficient random access.
//!
//! For a value *x* > 0, the number of stored bits is ⌊log₂ *x*⌋ (i.e.,
//! `bit_len(x) − 1`). To recover *x*, one reads the stored bits and
//! prepends a 1-bit.
//!
//! The structure implements [`SliceByValue`] for random access.
//!
//! # Examples
//!
//! ```rust
//! use sux::dict::comp_int_list::CompIntList;
//! use value_traits::slices::SliceByValue;
//!
//! let values = vec![1u64, 3, 7, 42, 100];
//! let list = CompIntList::new(values);
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
use crate::traits::bit_field_slice::Word;
use crate::utils::PrimitiveUnsignedExt;

/// Builder for creating a [`CompIntList`].
///
/// Values must be strictly positive and are pushed one at a time.
///
/// # Examples
///
/// ```rust
/// use sux::dict::comp_int_list::CompIntListBuilder;
/// use value_traits::slices::SliceByValue;
///
/// let mut builder = CompIntListBuilder::new();
/// builder.push(1u64);
/// builder.push(3);
/// builder.push(7);
/// builder.push(42);
///
/// let list = builder.build();
/// assert_eq!(list.index_value(0), 1);
/// assert_eq!(list.index_value(3), 42);
/// ```
#[derive(Debug, Clone, MemDbg, MemSize)]
pub struct CompIntListBuilder<V: Word = u64> {
    values: Vec<V>,
}

impl<V: Word> CompIntListBuilder<V> {
    /// Creates a new empty builder.
    pub fn new() -> Self {
        CompIntListBuilder { values: vec![] }
    }

    /// Pushes a strictly positive value.
    ///
    /// # Panics
    ///
    /// Panics if `value` is zero.
    pub fn push(&mut self, value: V) {
        assert!(value > V::ZERO, "CompIntList requires strictly positive values");
        self.values.push(value);
    }

    /// Builds the [`CompIntList`].
    pub fn build(self) -> CompIntList<V> {
        let values = self.values;
        let n = values.len();

        // Compute total bits needed
        let mut total_bits: u64 = 0;
        for &v in &values {
            total_bits += (v.bit_len() - 1) as u64;
        }

        // Build delimiters: n + 1 cumulative bit positions
        let mut efb = EliasFanoBuilder::new(n + 1, total_bits);
        let mut pos: u64 = 0;
        // SAFETY: pos = 0 is ≤ total_bits and is the first push
        unsafe { efb.push_unchecked(0u64) };
        for &v in &values {
            pos += (v.bit_len() - 1) as u64;
            // SAFETY: pos is non-decreasing and ≤ total_bits
            unsafe { efb.push_unchecked(pos) };
        }
        let delimiters = efb.build_with_seq();

        // Build data: pack the MSB-removed representations
        let n_words = (total_bits as usize).div_ceil(V::BITS as usize);
        let mut data = vec![V::ZERO; n_words];

        let mut bit_pos = 0usize;
        for &v in &values {
            let width = (v.bit_len() - 1) as usize;
            if width > 0 {
                // Strip the MSB: keep only the low `width` bits
                let bits = v ^ (V::ONE << width);
                CompIntList::<V>::write_bits(&mut data, bit_pos, bits, width);
            }
            bit_pos += width;
        }

        CompIntList {
            n,
            delimiters,
            data: data.into_boxed_slice(),
        }
    }
}

impl<V: Word> Default for CompIntListBuilder<V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V: Word> Extend<V> for CompIntListBuilder<V> {
    fn extend<I: IntoIterator<Item = V>>(&mut self, iter: I) {
        for v in iter {
            self.push(v);
        }
    }
}

/// A compact list of strictly positive integers.
///
/// Values are stored by concatenating their binary representations with the
/// most significant bit removed. The boundaries between values are recorded
/// in a [`SliceByValue<Value = u64>`] (by default an [Elias–Fano
/// sequence](EfSeq)), enabling efficient random access.
///
/// Use [`CompIntListBuilder`] for incremental construction, or [`CompIntList::new`]
/// to build from an iterator directly. After construction, the delimiter
/// structure can be replaced using [`map_delimiters`](CompIntList::map_delimiters).
///
/// # Type Parameters
///
/// - `V`: The value type. Must be a [`Word`] type. Defaults to `u64`.
/// - `D`: The delimiter structure. Must implement `SliceByValue<Value = u64>`.
///   Defaults to [`EfSeq`].
#[derive(Debug, Clone, MemDbg, MemSize)]
pub struct CompIntList<V: Word = u64, D: SliceByValue<Value = u64> = EfSeq> {
    /// Number of stored values.
    n: usize,
    /// Structure storing `n + 1` cumulative bit-position delimiters.
    delimiters: D,
    /// Concatenated binary representations (MSB removed), backed by
    /// `V`-sized words.
    data: Box<[V]>,
}

impl<V: Word> CompIntList<V> {
    /// Creates a new `CompIntList` from an iterator of strictly positive values.
    ///
    /// # Panics
    ///
    /// Panics if any value is zero.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sux::dict::comp_int_list::CompIntList;
    /// use value_traits::slices::SliceByValue;
    ///
    /// let ef = CompIntList::new(vec![1u64, 5, 10]);
    /// assert_eq!(ef.len(), 3);
    /// assert_eq!(ef.index_value(1), 5);
    /// ```
    pub fn new(values: impl IntoIterator<Item = V>) -> Self {
        let mut builder = CompIntListBuilder::new();
        builder.extend(values);
        builder.build()
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

impl<V: Word, D: SliceByValue<Value = u64>> SliceByValue for CompIntList<V, D> {
    type Value = V;

    #[inline(always)]
    fn len(&self) -> usize {
        self.n
    }

    #[inline(always)]
    unsafe fn get_value_unchecked(&self, index: usize) -> V {
        let start = unsafe { self.delimiters.get_value_unchecked(index) } as usize;
        let end = unsafe { self.delimiters.get_value_unchecked(index + 1) } as usize;
        let width = end - start;

        if width == 0 {
            return V::ONE;
        }

        let bits = unsafe { Self::read_bits(&self.data, start, width) };
        (V::ONE << width) | bits
    }
}
