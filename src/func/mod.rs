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
//! - [`VFunc`] is a generic static function.
//!
//! - [`LcpMmphfInt`]/[`LcpMmphf`] are *monotone minimal perfect hash
//!   functions*—specialized static functions mapping keys in lexicographical
//!   order to their lexicographical rank. See [`LcpMmphfStr`] and
//!   [`LcpMmphfSliceU8`] for common instantiations.

mod vfunc;
pub use vfunc::*;

#[cfg(feature = "rayon")]
mod vbuilder;
#[cfg(feature = "rayon")]
pub use vbuilder::*;

pub mod lcp_mmphf;
pub use lcp_mmphf::*;

pub mod signed_lcp_mmphf;
pub use signed_lcp_mmphf::*;

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
