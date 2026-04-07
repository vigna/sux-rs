/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! A compact list of nonnegative integers based on prefix sums.
//!
//! Given a list of nonnegative integers, a [`PrefixSumIntList`] stores the
//! cumulative (prefix) sums in a [`SliceByValue`] structure (by default an
//! [Elias–Fano sequence](crate::dict::elias_fano::EfSeq)). The original
//! values are recovered by taking differences of consecutive prefix sums.
//!
//! The prefix-sum structure can be [replaced](PrefixSumIntList::map_prefix_sums)
//! with any structure implementing [`IntoIteratorFrom`] with the returned
//! iterator implementing `UncheckedIterator<Item = D::Value>`, as long as it
//! returns the same cumulative sums.
//!
//! This structure implements [`SliceByValue`] for random access.
//!
//! # Examples
//!
//! ```rust
//! use sux::list::prefix_sum_int_list::PrefixSumIntList;
//! use value_traits::slices::SliceByValue;
//!
//! let values = vec![3u64, 1, 4, 1, 5];
//! let list = PrefixSumIntList::new(&values);
//!
//! assert_eq!(list.index_value(0), 3);
//! assert_eq!(list.index_value(1), 1);
//! assert_eq!(list.index_value(2), 4);
//! assert_eq!(list.index_value(3), 1);
//! assert_eq!(list.index_value(4), 5);
//!
//! // Prefix sums (exclusive)
//! assert_eq!(list.prefix_sum(0), 0);
//! assert_eq!(list.prefix_sum(1), 3);
//! assert_eq!(list.prefix_sum(2), 4);
//! assert_eq!(list.prefix_sum(5), 14);
//! ```

use mem_dbg::*;
use value_traits::slices::SliceByValue;

use crate::dict::elias_fano::EfSeq;
use crate::dict::{EliasFano, EliasFanoBuilder};
use crate::traits::iter::{IntoIteratorFrom, UncheckedIterator};
use crate::traits::{SelectUnchecked, Word};

/// A compact list of nonnegative integers based on prefix sums.
///
/// The original values are stored implicitly as differences of consecutive
/// prefix sums held in a [`SliceByValue`] structure (by default an
/// [Elias–Fano sequence](EfSeq)). Recovering the *i*-th value requires two
/// accesses to the prefix-sum structure.
///
/// The structure provides also [direct access to the prefix
/// sums](PrefixSumIntList::prefix_sum).
///
/// After construction, the prefix-sum structure can be replaced using
/// [`map_prefix_sums`](PrefixSumIntList::map_prefix_sums).
///
/// # Type Parameters
///
/// - `D`: The prefix-sum structure. Must implement `SliceByValue` with a
///   [`Word`] value type. Defaults to [`EfSeq<usize>`](EfSeq).
#[derive(Debug, Clone, MemDbg, MemSize)]
pub struct PrefixSumIntList<D = EfSeq<usize>> {
    /// Number of stored values.
    ///
    /// Note that this is identical to `prefix_sums.len() - 1`, but we have no
    /// guarantee that calling `prefix_sums.len()` is O(1).
    n: usize,
    /// Structure storing `n + 1` monotone prefix sums.
    prefix_sums: D,
}

impl<V: Word> PrefixSumIntList<EfSeq<V>> {
    /// Creates a new `PrefixSumIntList` from a reference to a collection of
    /// nonnegative values.
    ///
    /// The collection is iterated twice: once to compute the total sum, and
    /// once to build the prefix-sum structure.
    ///
    /// # Panics
    ///
    /// Panics if the prefix sums overflow `V`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use sux::list::prefix_sum_int_list::PrefixSumIntList;
    /// # use value_traits::slices::SliceByValue;
    /// let values = vec![3u64, 1, 4, 1, 5];
    /// let list = PrefixSumIntList::new(&values);
    /// assert_eq!(list.len(), 5);
    /// assert_eq!(list.index_value(2), 4);
    /// assert_eq!(list.prefix_sum(5), 14);
    /// ```
    pub fn new<I: ?Sized>(values: &I) -> Self
    where
        for<'a> &'a I: IntoIterator<Item = &'a V>,
    {
        // First pass: count elements and total sum
        let mut n = 0;
        let mut total = V::ZERO;
        for &v in values {
            n += 1;
            total = total
                .checked_add(v)
                .expect("PrefixSumIntList: prefix sum overflow");
        }

        // Second pass: build prefix sums in Elias–Fano
        let mut efb = EliasFanoBuilder::new(n + 1, total);
        let mut prefix = V::ZERO;
        // SAFETY: prefix = 0 ≤ total and is the first push
        unsafe { efb.push_unchecked(prefix) };
        for &v in values {
            // Cannot overflow
            prefix += v;
            // SAFETY: prefix is non-decreasing and ≤ total
            unsafe { efb.push_unchecked(prefix) };
        }
        let prefix_sums = efb.build_with_seq();

        PrefixSumIntList { n, prefix_sums }
    }
}

/// Creates a `PrefixSumIntList` from an existing Elias—Fano structure.
impl<V: Word, H: AsRef<[usize]> + SelectUnchecked, L: SliceByValue<Value = V>>
    From<EliasFano<V, H, L>> for PrefixSumIntList<EliasFano<V, H, L>>
{
    fn from(elias_fano: EliasFano<V, H, L>) -> Self {
        PrefixSumIntList {
            n: elias_fano.len() - 1,
            prefix_sums: elias_fano,
        }
    }
}

impl<D: SliceByValue<Value: Word>> PrefixSumIntList<D> {
    /// Creates a `PrefixSumIntList` from an existing prefix-sum structure.
    ///
    /// The structure must contain at least one element (the initial zero),
    /// its first element must be zero, and its values must be monotonically
    /// nondecreasing.
    ///
    /// Returns an error string if any of these conditions is violated.
    pub fn try_from_prefix_sums(prefix_sums: D) -> Result<Self, &'static str> {
        let len = prefix_sums.len();
        if len == 0 {
            return Err("PrefixSumIntList: prefix-sum sequence must be non-empty");
        }
        let mut prev = prefix_sums.index_value(0);
        if prev != D::Value::ZERO {
            return Err("PrefixSumIntList: first element must be zero");
        }
        for i in 1..len {
            let cur = prefix_sums.index_value(i);
            if cur < prev {
                return Err("PrefixSumIntList: values must be monotonically nondecreasing");
            }
            prev = cur;
        }
        Ok(PrefixSumIntList {
            n: len - 1,
            prefix_sums,
        })
    }
}

impl<D: SliceByValue> PrefixSumIntList<D> {
    /// Replaces the prefix-sum structure.
    ///
    /// # Safety
    ///
    /// This method is unsafe because it is not possible to guarantee that the
    /// new structure returns the same values as the old one.
    pub unsafe fn map_prefix_sums<F, D2>(self, func: F) -> PrefixSumIntList<D2>
    where
        F: FnOnce(D) -> D2,
    {
        PrefixSumIntList {
            n: self.n,
            prefix_sums: func(self.prefix_sums),
        }
    }

    /// Returns the underlying prefix-sum structure.
    pub fn into_inner(self) -> D {
        self.prefix_sums
    }
}

impl<D: SliceByValue> PrefixSumIntList<D>
where
    for<'a> &'a D: IntoIteratorFrom,
    for<'a> <&'a D as IntoIteratorFrom>::IntoIterFrom: UncheckedIterator<Item = D::Value>,
{
    /// Returns the prefix sum up to (excluded) index `i`.
    ///
    /// `prefix_sum(0)` is 0, and `prefix_sum(n)` is the sum of all values.
    ///
    /// # Panics
    ///
    /// Panics if `i > self.len()`.
    #[inline(always)]
    pub fn prefix_sum(&self, i: usize) -> D::Value {
        self.prefix_sums.index_value(i)
    }

    /// Returns the prefix sum up to (excluded) index `i` without bounds
    /// checking.
    ///
    /// # Safety
    ///
    /// `i` must be in `0..=self.len()`.
    #[inline(always)]
    pub unsafe fn prefix_sum_unchecked(&self, i: usize) -> D::Value {
        unsafe { self.prefix_sums.get_value_unchecked(i) }
    }
}

impl<D: SliceByValue<Value: Word>> SliceByValue for PrefixSumIntList<D>
where
    for<'a> &'a D: IntoIteratorFrom,
    for<'a> <&'a D as IntoIteratorFrom>::IntoIterFrom: UncheckedIterator<Item = D::Value>,
{
    type Value = D::Value;

    #[inline(always)]
    fn len(&self) -> usize {
        self.n
    }

    #[inline]
    unsafe fn get_value_unchecked(&self, index: usize) -> D::Value {
        let mut iter = (&self.prefix_sums).into_iter_from(index);
        let start = unsafe { iter.next_unchecked() };
        let end = unsafe { iter.next_unchecked() };
        end - start
    }
}
