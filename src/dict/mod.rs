/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! (Indexed) dictionaries.
//!
//! # Membership structures
//!
//! - [`VFilter`]: Static filters (approximate membership structures) based on
//!   hash comparison. Use the `Box<[W]>` backend for full-width hashes, or the
//!   `BitFieldVec` backend with an explicit `filter_bits` parameter for
//!   space/false-positive tradeoffs.
//!
//! # Indexed dictionaries
//!
//! An indexed dictionary maps [integer indices to values], possibly
//! supporting [lookups] and [predecessor]/[successor] queries. The main
//! structures are:
//!
//! - [`EliasFano`]: A compact representation of monotone integer sequences,
//!   supporting efficient access, successor, and predecessor queries.
//! - [`RearCodedListStr`] / [`RearCodedListSliceU8`]: Prefix-compressed
//!   immutable lists of strings or byte sequences with random access.
//! - [`MappedRearCodedListStr`] / [`MappedRearCodedListSliceU8`]: Rear-coded
//!   lists with element reordering for better compression.
//! - [`SignedFunc`]: [Functions] verified by hash signatures.
//! - [`SliceSeq`]: Adapters exposing slice references as indexed sequences.
//!
//! Most structures implement the [`TryIntoUnaligned`] trait, allowing them
//! to be converted into (usually faster) structures using unaligned access.
//!
//! [integer indices to values]: crate::traits::IndexedSeq
//! [lookups]: crate::traits::IndexedDict
//! [predecessor]: crate::traits::Pred
//! [successor]: crate::traits::Succ
//! [`SignedFunc`]: crate::func::signed::SignedFunc
//! [Functions]: crate::func
//! [`TryIntoUnaligned`]: crate::traits::TryIntoUnaligned

pub mod elias_fano;
pub use elias_fano::{
    EfDict, EfSeq, EfSeqDict, EliasFano, EliasFanoBuilder, EliasFanoConcurrentBuilder,
};

pub mod rear_coded_list;
pub use rear_coded_list::{RearCodedListBuilder, RearCodedListSliceU8, RearCodedListStr};

pub mod mapped_rear_coded_list;
pub use mapped_rear_coded_list::{MappedRearCodedListSliceU8, MappedRearCodedListStr};

mod slice_seq;
pub use slice_seq::SliceSeq;

pub mod part_elias_fano;
pub use part_elias_fano::{PartEliasFano, PartEliasFanoBuilder};

pub mod flat_part_elias_fano;
pub use flat_part_elias_fano::{FlatPartEliasFano, FlatPartEliasFanoBuilder};

pub mod vfilter;
pub use vfilter::VFilter;
