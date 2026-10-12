/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

//! Static functions, minimal perfect hash functions, and monotone minimal
//! perfect hash functions.
//!
//! Static functions map keys to values, but they do not store the keys:
//! querying a static function with a key outside of the original set will lead
//! to an arbitrary result.
//!
//! In exchange, static functions make it possible to store the association
//! between keys and values just in the space required by the values, plus
//! a small overhead.
//!
//! # Choosing a structure
//!
//! | You need to… | Use |
//! |---|---|
//! | associate a value with each key | [`VFunc`]; [`VFunc2`] or [`CompVFunc`] if the distribution of the values is skewed |
//! | map the keys bijectively to [0 . . *n*), in any order | [`PHastR`] |
//! | map the keys to their rank in lexicographical order | [`LcpMmphf`]/[`LcpMmphfInt`], or [`Lcp2Mmphf`]/[`Lcp2MmphfInt`] to use less space |
//! | reject keys outside the original set | a [`SignedFunc`] wrapping one of the above |
//! | test approximate membership only | a [`VFilter`] |
//!
//! - [`VFunc`] is a generic static function.
//!
//! - [`VFunc2`] is a two-step static function that has better space usage
//!   in case the distribution of the output values is skewed.
//!
//! - [`CompVFunc`] is a compressed static function that stores values in
//!   space close to the zero-order entropy of their distribution using a
//!   Huffman code.
//!
//! - [`LcpMmphfInt`]/[`LcpMmphf`] are *monotone minimal perfect hash
//!   functions*, specialized static functions mapping keys in lexicographical
//!   order to their lexicographical rank. See [`LcpMmphfStr`] and
//!   [`LcpMmphfSliceU8`] for common instantiations.
//!
//! - [`Lcp2MmphfInt`]/[`Lcp2Mmphf`] are versions of
//!   [`LcpMmphfInt`]/[`LcpMmphf`] that use a [`VFunc2`]-like technique to reduce
//!   space usage, at the cost of slightly slower queries.
//!
//! - [`PHastR`] is a *minimal perfect hash function*, mapping bijectively
//!   keys to their number in an arbitrary order, with fast construction and
//!   queries.
//!
//! - [`SignedFunc`] wraps a [`VFunc`], a [`PHastR`],
//!   [`LcpMmphfInt`]/[`LcpMmphf`], or [`Lcp2MmphfInt`]/[`Lcp2Mmphf`] (but not
//!   [`VFunc2`] or [`CompVFunc`]) with per-key verification hashes, returning
//!   `None` for keys outside the original set. Use `Box<[W]>` for full-width
//!   hashes or [`BitFieldVec`] for sub-word-width hashes. Use concrete types
//!   like `SignedFunc<LcpMmphfStr, Box<[u64]>>`; the [supported
//!   combinations](SignedFunc#supported-combinations) are listed in the
//!   documentation of [`SignedFunc`].
//!
//! # Construction
//!
//! All structures provide four constructors:
//!
//! - `try_par_new(keys, …, pl)` builds the structure from slices of keys (and
//!   values), computing the hashes of the keys in parallel: use it when the
//!   keys fit in memory;
//!
//! - `try_new(keys, …, pl)` reads the keys (and values) from a [rewindable
//!   lender], such as the lines of a file, without keeping them in memory;
//!   the [`lenders`] module explains how to obtain one;
//!
//! - `try_par_new_with_builder` and `try_new_with_builder` take an additional
//!   builder that configures the construction: a [`VBuilder`] (offline mode,
//!   number of threads, sharding overhead, seed) or, for [`PHastR`], a
//!   [`PHastRBuilder`].
//!
//! The functions based on longest common prefixes also need the number of
//! keys, and their keys must be sorted. Building a structure requires the
//! `rayon` feature, except for [`PHastR`], which can be built sequentially
//! without it.
//!
//! Most structures implement the [`TryIntoUnaligned`] trait, allowing them
//! to be converted into (usually faster) structures using unaligned access.
//! With the `epserde` feature, all structures can be serialized with
//! [ε-serde], and then loaded or memory-mapped.
//!
//! # Command-line utilities
//!
//! With the `cli` feature, the binaries `vfunc`, `vfunc2`, `comp_vfunc`,
//! `vfilter`, `lcp_mmphf`, and `phastr` build these structures from files of
//! keys and serialize them with ε-serde, logging the type of the structure
//! they store; use `--help` for their options.
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
//!
//! [`BitFieldVec`]: crate::bits::BitFieldVec
//! [`TryIntoUnaligned`]: crate::traits::TryIntoUnaligned
//! [`VFilter`]: crate::dict::VFilter
//! [rewindable lender]: crate::utils::FallibleRewindableLender
//! [`lenders`]: crate::utils::lenders
//! [ε-serde]: https://crates.io/crates/epserde

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
#[doc(inline)]
pub use lcp_mmphf::*;

pub mod lcp2_mmphf;
#[doc(inline)]
pub use lcp2_mmphf::*;

pub mod signed;
#[doc(inline)]
pub use signed::*;

pub mod shard_edge;

pub mod codec;

mod comp_vfunc;
pub use comp_vfunc::CompVFunc;

pub mod phastr;
#[doc(inline)]
pub use phastr::{PHastR, PHastRBuilder};
