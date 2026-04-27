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
//! - [`VFunc2`] is a two-step static function that has better space usage
//!   in case the distribution of the output values is skewed.
//!
//! - [`LcpMmphfInt`]/[`LcpMmphf`] are *monotone minimal perfect hash
//!   functions*—specialized static functions mapping keys in lexicographical
//!   order to their lexicographical rank. See [`LcpMmphfStr`] and
//!   [`LcpMmphfSliceU8`] for common instantiations.
//!
//! - [`Lcp2MmphfInt`]/[`Lcp2Mmphf`] are versions of
//!   [`LcpMmphfInt`]/[`LcpMmphf`] that use a [`VFunc2`]-like technique to reduce
//!   space usage, at the cost of slightly slower queries.
//!
//! - [`SignedFunc`] wraps any of the above with per-key verification hashes,
//!   returning `None` for keys outside the original set. Use `Box<[W]>` for
//!   full-width hashes or [`BitFieldVec`] for sub-word-width hashes. Use
//!   concrete types like `SignedFunc<LcpMmphfStr, Box<[u64]>>`.
//!
//! Most structures implement the [`TryIntoUnaligned`] trait, allowing them
//! to be converted into (usually faster) structures using unaligned access.
//!
//! [`BitFieldVec`]: crate::bits::BitFieldVec
//! [`TryIntoUnaligned`]: crate::traits::TryIntoUnaligned
//!
//! All constructors follow the pattern `try_new(keys, …, pl)` for default
//! settings, and `try_new_with_builder(keys, …, builder, pl)` to configure
//! the [`VBuilder`] (offline mode, thread count, sharding overhead, seed).
//!
//! # Type annotations
//!
//! Functions have a formidable number of parameters, many of which have
//! defaults. However, Rust does not apply struct default type parameters in
//! expression position, so constructor calls like `VFunc::try_new` will
//! not work as they cannot infer the default arguments.
//!
//! The fix is to write the type between angular brackets in the constructor
//! call, which applies defaults:
//!
//! ```rust
//! # #[cfg(feature = "rayon")]
//! # fn main() -> anyhow::Result<()> {
//! # use sux::func::VFunc;
//! # use dsi_progress_logger::no_logging;
//! # use sux::utils::FromCloneableIntoIterator;
//! let func = <VFunc<usize, Box<[u8]>>>::try_new(
//!     FromCloneableIntoIterator::new(0..10_usize),
//!     FromCloneableIntoIterator::new(0..10_u8),
//!     no_logging![],
//! )?;
//! # Ok(())
//! # }
//! # #[cfg(not(feature = "rayon"))]
//! # fn main() {}
//! ```

mod vfunc;
pub use vfunc::*;

mod vfunc2;
pub use vfunc2::*;

#[cfg(feature = "rayon")]
pub(crate) mod peeling;

#[cfg(feature = "rayon")]
mod vbuilder;
#[cfg(feature = "rayon")]
pub use vbuilder::*;

pub mod lcp_mmphf;
pub use lcp_mmphf::*;

pub mod lcp2_mmphf;
pub use lcp2_mmphf::*;

pub mod signed;
pub use signed::*;

pub mod shard_edge;

pub mod codec;

#[cfg(feature = "rayon")]
pub mod comp_vfunc;
#[cfg(feature = "rayon")]
pub use comp_vfunc::CompVFunc;

/// Avalanches bits using the finalization step of Austin Appleby's
/// [MurmurHash3].
///
/// [MurmurHash3]: http://code.google.com/p/smhasher/
pub(crate) const fn mix64(mut k: u64) -> u64 {
    k ^= k >> 33;
    k = k.overflowing_mul(0xff51_afd7_ed55_8ccd).0;
    k ^= k >> 33;
    k = k.overflowing_mul(0xc4ce_b9fe_1a85_ec53).0;
    k ^= k >> 33;
    k
}
