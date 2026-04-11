/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! A compact list of nonnegative integers based on prefix sums.
//!
//! Given a list of nonnegative integers, a [`PrefixSumIntList`] stores the
//! cumulative (prefix) sums in a [`SliceByValue`] structure (by default an
//! [Elias–Fano sequence]). The original
//! values are recovered by taking differences of consecutive prefix sums.
//!
//! The prefix-sum structure can be [replaced] with any structure
//! implementing [`IntoIteratorFrom`] with the returned iterator
//! implementing `UncheckedIterator<Item = usize>`, as long as it returns
//! the same cumulative sums.
//!
//! This structure implements [`SliceByValue`] for random access.
//!
//! # Examples
//!
//! ```rust
//! use sux::list::prefix_sum_int_list::PrefixSumIntList;
//! use value_traits::slices::SliceByValue;
//!
//! let values = vec![3usize, 1, 4, 1, 5];
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
//!
//! [Elias–Fano sequence]: crate::dict::elias_fano::EfSeq
//! [replaced]: PrefixSumIntList::map_prefix_sums

use mem_dbg::*;
use value_traits::slices::SliceByValue;

use crate::dict::elias_fano::EfSeq;
use crate::dict::{EliasFano, EliasFanoBuilder};
use crate::traits::SelectUnchecked;
use crate::traits::iter::{IntoIteratorFrom, UncheckedIterator};

/// A compact list of nonnegative integers based on prefix sums.
///
/// The original values are stored implicitly as differences of consecutive
/// prefix sums held in a [`SliceByValue<Value = usize>`] (by default an
/// [Elias–Fano sequence]). Recovering the *i*-th value requires two
/// accesses to the prefix-sum structure.
///
/// The structure provides also [direct access to the prefix
/// sums].
///
/// After construction, the prefix-sum structure can be replaced using
/// [`map_prefix_sums`].
///
/// This structure implements the [`TryIntoUnaligned`]
/// trait, allowing it to be converted into (usually faster) structures using
/// unaligned access.
///
/// # Type Parameters
///
/// - `D`: The prefix-sum structure. Must implement `SliceByValue<Value = usize>`.
///   Defaults to [`EfSeq<usize>`].
///
/// [Elias–Fano sequence]: EfSeq
/// [direct access to the prefix sums]: PrefixSumIntList::prefix_sum
/// [`map_prefix_sums`]: PrefixSumIntList::map_prefix_sums
/// [`EfSeq<usize>`]: EfSeq
#[derive(Debug, Clone, MemSize, MemDbg)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PrefixSumIntList<D = EfSeq<usize>> {
    /// Number of stored values.
    ///
    /// Note that this is identical to `prefix_sums.len() - 1`, but we have no
    /// guarantee that calling `prefix_sums.len()` is O(1).
    n: usize,
    /// Structure storing `n + 1` monotone prefix sums.
    prefix_sums: D,
}

impl PrefixSumIntList {
    /// Creates a new `PrefixSumIntList` from a reference to a collection of
    /// nonnegative values.
    ///
    /// The collection is iterated twice: once to compute the total sum, and
    /// once to build the prefix-sum structure.
    ///
    /// # Panics
    ///
    /// Panics if the prefix sums overflow `usize`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use sux::list::prefix_sum_int_list::PrefixSumIntList;
    /// # use value_traits::slices::SliceByValue;
    /// let values = vec![3usize, 1, 4, 1, 5];
    /// let list = PrefixSumIntList::new(&values);
    /// assert_eq!(list.len(), 5);
    /// assert_eq!(list.index_value(2), 4);
    /// assert_eq!(list.prefix_sum(5), 14);
    /// ```
    pub fn new<I: ?Sized>(values: &I) -> Self
    where
        for<'a> &'a I: IntoIterator<Item = &'a usize>,
    {
        // First pass: count elements and total sum
        let mut n = 0;
        let mut total: usize = 0;
        for &v in values {
            n += 1;
            total = total
                .checked_add(v)
                .expect("PrefixSumIntList: prefix sum overflow");
        }

        // Second pass: build prefix sums in Elias–Fano
        let mut efb = EliasFanoBuilder::new(n + 1, total);
        let mut prefix: usize = 0;
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

/// Creates a `PrefixSumIntList` from an existing Elias--Fano structure.
impl<H: AsRef<[usize]> + SelectUnchecked, L: SliceByValue<Value = usize>>
    From<EliasFano<usize, H, L>> for PrefixSumIntList<EliasFano<usize, H, L>>
{
    fn from(elias_fano: EliasFano<usize, H, L>) -> Self {
        PrefixSumIntList {
            n: elias_fano.len() - 1,
            prefix_sums: elias_fano,
        }
    }
}

impl<D: SliceByValue<Value = usize>> PrefixSumIntList<D> {
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
        if prev != 0 {
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

impl<D: SliceByValue<Value = usize>> PrefixSumIntList<D> {
    /// Replaces the prefix-sum structure.
    ///
    /// # Safety
    ///
    /// This method is unsafe because it is not possible to guarantee that the
    /// new structure returns the same values as the old one.
    pub unsafe fn map_prefix_sums<F, D2>(self, func: F) -> PrefixSumIntList<D2>
    where
        F: FnOnce(D) -> D2,
        D2: SliceByValue<Value = usize>,
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

impl<D: SliceByValue<Value = usize>> PrefixSumIntList<D>
where
    for<'a> &'a D: IntoIteratorFrom,
    for<'a> <&'a D as IntoIteratorFrom>::IntoIterFrom: UncheckedIterator<Item = usize>,
{
    /// Returns the prefix sum up to (excluded) index `i`.
    ///
    /// `prefix_sum(0)` is 0, and `prefix_sum(n)` is the sum of all values.
    ///
    /// # Panics
    ///
    /// Panics if `i > self.len()`.
    #[inline(always)]
    pub fn prefix_sum(&self, i: usize) -> usize {
        self.prefix_sums.index_value(i)
    }

    /// Returns the prefix sum up to (excluded) index `i` without bounds
    /// checking.
    ///
    /// # Safety
    ///
    /// `i` must be in `0..=self.len()`.
    #[inline(always)]
    pub unsafe fn prefix_sum_unchecked(&self, i: usize) -> usize {
        unsafe { self.prefix_sums.get_value_unchecked(i) }
    }
}

impl<D: SliceByValue<Value = usize>> SliceByValue for PrefixSumIntList<D>
where
    for<'a> &'a D: IntoIteratorFrom,
    for<'a> <&'a D as IntoIteratorFrom>::IntoIterFrom: UncheckedIterator<Item = usize>,
{
    type Value = usize;

    #[inline(always)]
    fn len(&self) -> usize {
        self.n
    }

    #[inline]
    unsafe fn get_value_unchecked(&self, index: usize) -> usize {
        let mut iter = (&self.prefix_sums).into_iter_from(index);
        let start = unsafe { iter.next_unchecked() };
        let end = unsafe { iter.next_unchecked() };
        end - start
    }
}

use crate::traits::TryIntoUnaligned;

impl<D: TryIntoUnaligned + SliceByValue<Value = usize>> TryIntoUnaligned for PrefixSumIntList<D>
where
    D::Unaligned: SliceByValue<Value = usize>,
{
    type Unaligned = PrefixSumIntList<D::Unaligned>;
    fn try_into_unaligned(
        self,
    ) -> Result<Self::Unaligned, crate::traits::UnalignedConversionError> {
        Ok(PrefixSumIntList {
            n: self.n,
            prefix_sums: self.prefix_sums.try_into_unaligned()?,
        })
    }
}
