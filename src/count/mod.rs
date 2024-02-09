/*
 * SPDX-FileCopyrightText: 2023 Tommaso Fontana
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Probabilistic Counters
//! These are used to estimate the cardinality of unique elements of a stream.

mod hyperloglog_vec;
pub use hyperloglog_vec::{AtomicHyperLogLogVec, DefaultStrategy, HashStrategy, HyperLogLogVec};
