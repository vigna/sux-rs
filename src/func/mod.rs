/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Static functions.
//!
//! Static functions map keys to values, but they do not store the keys:
//! querying a static function with a key outside of the original set will lead
//! to an arbitrary result.
//!
//! In exchange, static functions make it possible to store the association
//! between keys and values just in the space required by the values, plus
//! a small overhead.
//!
//! See [`VFunc`] for more details.

mod vfunc;
pub use vfunc::*;

mod vbuilder;
pub use vbuilder::*;

pub mod shard_edge;

/// Avalanches bits using the finalization step of Austin Appleby's
/// [MurmurHash3](http://code.google.com/p/smhasher/).
#[doc(hidden)]
pub const fn mix64(mut k: u64) -> u64 {
    k ^= k >> 33;
    k = k.overflowing_mul(0xff51_afd7_ed55_8ccd).0;
    k ^= k >> 33;
    k = k.overflowing_mul(0xc4ce_b9fe_1a85_ec53).0;
    k ^= k >> 33;
    k
}
