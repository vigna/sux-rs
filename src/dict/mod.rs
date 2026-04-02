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
//! - [`SignedFunc`] / [`BitSignedFunc`]:
//!   Index functions verified by hash signatures, returning `None` on mismatch.
//!   These are re-exported from [`func::signed`](crate::func::signed); see
//!   also the type aliases [`SignedLcpMmphfInt`](crate::func::SignedLcpMmphfInt),
//!   [`SignedLcpMmphfStr`](crate::func::SignedLcpMmphfStr), etc.
//! - [`VFilter`]: Static filters (approximate membership structures) based on
//!   hash comparison. Use the `Box<[W]>` backend for full-width hashes, or the
//!   `BitFieldVec` backend with an explicit `filter_bits` parameter for
//!   space/false-positive tradeoffs. See the [choosing a type](crate::func#choosing-a-type)
//!   guide.
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

pub use crate::func::signed::{BitSignedFunc, SignedFunc};

pub mod vfilter;
pub use vfilter::VFilter;

use crate::{func::shard_edge::ShardEdge, utils::Sig};

/// Common interface for inner minimal perfect hash functions used by signed
/// wrappers.
///
/// Provides access to the seed, shard edge, and key count, so that
/// [`SignedFunc`](crate::func::SignedFunc) and
/// [`BitSignedFunc`](crate::func::BitSignedFunc) can verify hashes
/// without knowing which specific MMPHF variant they wrap.
pub trait SignableMphf {
    type Sig: Sig;
    type Edge: ShardEdge<Self::Sig, 3>;

    fn seed(&self) -> u64;
    fn shard_edge(&self) -> &Self::Edge;
    fn len(&self) -> usize;
    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
