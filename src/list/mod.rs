/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Compact representations of lists.
//!
//! This module contains compact representations of lists. Note that [(indexed)
//! dictionaries](crate::dict) can sometimes be used to represent lists of
//! distinct elements, but implementations in this module do not have this
//! restriction, as they implement just
//! [`SliceByValue`](value_traits::slices::SliceByValue).

pub mod comp_int_list;
pub use comp_int_list::CompIntList;

pub mod prefix_sum_int_list;
pub use prefix_sum_int_list::PrefixSumIntList;
