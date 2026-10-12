/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

//! Indexed dictionaries and static filters.
//!
//! # Membership structures
//!
//! - [`VFilter`]: Static filters (approximate membership structures) based on
//!   hash comparison. Use the `Box<[W]>` backend for full-width hashes, or the
//!   `BitFieldVec` backend with an explicit `filter_bits` parameter for
//!   space/false-positive tradeoffs.
//! - [`SignedFunc`] (in the [`func`] module): [Functions] verified by hash
//!   signatures, returning `None` (except for false positives) on keys outside
//!   the original set.
//!
//! # Indexed dictionaries
//!
//! An indexed dictionary maps [integer indices to values], possibly
//! supporting [lookups] and [predecessor]/[successor] queries. The main
//! structures are:
//!
//! - [`EliasFano`]: A compact representation of monotone integer sequences,
//!   supporting efficient access, successor, and predecessor queries. You
//!   build it with an [`EliasFanoBuilder`] (sequentially) or an
//!   [`EliasFanoConcurrentBuilder`] (concurrently); the type aliases
//!   [`EfSeq`], [`EfDict`], and [`EfSeqDict`] are the default types supporting
//!   access, dictionary operations, or both.
//! - [`RearCodedListStr`] / [`RearCodedListSliceU8`]: Prefix-compressed
//!   immutable lists of strings or byte sequences with random access, built
//!   using a [`RearCodedListBuilder`].
//! - [`MappedRearCodedListStr`] / [`MappedRearCodedListSliceU8`]: Rear-coded
//!   lists with element reordering for better compression.
//! - [`SliceSeq`]: Adapters exposing slice references as indexed sequences.
//!
//! The [`EliasFano`] and [`VFilter`] structures implement the
//! [`TryIntoUnaligned`] trait, allowing them to be converted into (usually
//! faster) structures using unaligned access.
//!
//! [integer indices to values]: crate::traits::IndexedSeq
//! [lookups]: crate::traits::IndexedDict
//! [predecessor]: crate::traits::Pred
//! [successor]: crate::traits::Succ
//! [`SignedFunc`]: crate::func::SignedFunc
//! [`func`]: crate::func
//! [Functions]: crate::func
//! [`TryIntoUnaligned`]: crate::traits::TryIntoUnaligned

pub mod elias_fano;
#[doc(inline)]
pub use elias_fano::{
    EfDict, EfSeq, EfSeqDict, EliasFano, EliasFanoBuilder, EliasFanoConcurrentBuilder,
};

pub mod rear_coded_list;
#[doc(inline)]
pub use rear_coded_list::{RearCodedListBuilder, RearCodedListSliceU8, RearCodedListStr};

pub mod mapped_rear_coded_list;
#[doc(inline)]
pub use mapped_rear_coded_list::{MappedRearCodedListSliceU8, MappedRearCodedListStr};

mod slice_seq;
pub use slice_seq::SliceSeq;

pub mod vfilter;
#[doc(inline)]
pub use vfilter::VFilter;
