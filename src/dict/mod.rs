/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Indexed dictionaries.
//!
//! An indexed dictionary maps integer indices to values, supporting both
//! random access and predecessor/successor queries. The main structures are:
//!
//! - [`EliasFano`]: A compact representation of monotone integer sequences,
//!   supporting efficient access, successor, and predecessor queries.
//! - [`RearCodedListStr`] / [`RearCodedListSliceU8`]: Prefix-compressed
//!   immutable lists of strings or byte sequences with random access.
//! - [`MappedRearCodedListStr`] / [`MappedRearCodedListSliceU8`]: Rear-coded
//!   lists with element reordering for better compression.
//! - [`SignedVFunc`] / [`BitSignedVFunc`]:
//!   Index functions verified by hash signatures, returning `None` on mismatch.
//! - [`VFilter`]: Static filters (approximate membership structures) based on
//!   hash comparison.
//! - [`SliceSeq`]: Adapters exposing slice references as indexed sequences.
//!
//! These structures implement traits from the
//! [`indexed_dict`](crate::traits::indexed_dict) module.
//!
//! Most structures implement the
//! [`TryIntoUnaligned`](crate::traits::TryIntoUnaligned) trait, allowing them
//! to be converted into (usually faster) structures using unaligned access.

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

pub mod signed_vfunc;
pub use signed_vfunc::{BitSignedVFunc, SignedVFunc};

pub mod vfilter;
pub use vfilter::VFilter;
