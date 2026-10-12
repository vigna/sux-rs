/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

//! Compact representations of lists.
//!
//! This module contains compact representations of lists of integers. All
//! structures implement [`SliceByValue`].
//!
//! - [`CompIntList`] stores each value *x* not less than a lower bound `min`
//!   using the binary representation of *x* − `min` + 1 without its most
//!   significant bit, and records the boundaries between these variable-length
//!   codes in an [Elias–Fano] structure. Each value uses ⌊log₂(*x* − `min` +
//!   1)⌋ bits, plus about 2 + log₂ of the average code length for its
//!   boundary, so this structure is a good choice for skewed distributions, in
//!   which most values are small but a few are large.
//!
//! - [`PrefixSumIntList`] stores the prefix sums of the values in an
//!   [Elias–Fano] structure, using about 2 + log₂ of the average value bits per
//!   value, independently of the distribution of the values. It is a good
//!   choice when values have similar magnitudes, and it provides access to
//!   [prefix sums].
//!
//! [`SliceByValue`]: value_traits::slices::SliceByValue
//! [Elias–Fano]: crate::dict::EliasFano
//! [prefix sums]: PrefixSumIntList::prefix_sum

pub mod comp_int_list;
#[doc(inline)]
pub use comp_int_list::CompIntList;

pub mod prefix_sum_int_list;
#[doc(inline)]
pub use prefix_sum_int_list::PrefixSumIntList;
