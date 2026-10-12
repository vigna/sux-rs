/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

#![doc = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/README.md"))]
#![cfg_attr(
    all(feature = "aarch64_prefetch", target_arch = "aarch64"),
    feature(stdarch_aarch64_prefetch)
)]
#![cfg_attr(feature = "iter_advance_by", feature(iter_advance_by))]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, doc(auto_cfg))]
#![deny(unconditional_recursion)]
#![allow(clippy::duplicated_attributes)]
#![allow(clippy::len_without_is_empty)]
#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]

pub mod array;
pub mod bal_paren;
pub mod bits;
pub mod dict;
pub mod func;
pub mod list;
pub mod rank_sel;
pub mod traits;
pub mod utils;

#[cfg(feature = "cli")]
pub mod cli;

// Kept for backward compatibility
#[cfg(feature = "cli")]
#[doc(hidden)]
pub use cli::init_env_logger;

#[cfg(feature = "fuzz")]
pub mod fuzz;

/// Imports the most common items.
///
/// The prelude contains all the structures of the crate, with their
/// builders and type aliases, the [`bit_vec!`], [`bit_field_vec!`], and
/// [`rank_small!`] macros, the traits needed to use bit vectors
/// ([`BitLength`], [`BitVecOps`], [`BitVecOpsMut`], [`AtomicBitVecOps`]), the
/// rank/select traits, the iteration traits, the [`TryIntoUnaligned`] trait
/// and the [`Unaligned`] type alias, the [`Word`] and [`Backend`] traits, and
/// the [lenders] most commonly passed to constructors ([`FromSlice`],
/// [`FromCloneableIntoIterator`], [`LineLender`]).
///
/// The [`bit_field_slice`] and [`indexed_dict`] module names are re-exported so
/// their items can be reached through the prelude, but their contents are not
/// glob-imported, as that would cause ambiguities in some contexts (e.g.,
/// [`IndexedSeq::get`] would shadow [`slice::get`]). To use bit-field vectors
/// and indexed dictionaries, add
///
/// ```rust
/// use sux::prelude::{*, bit_field_slice::*, indexed_dict::*};
/// ```
///
/// [`bit_field_slice`]: crate::traits::bit_field_slice
/// [`indexed_dict`]: crate::traits::indexed_dict
/// [`IndexedSeq::get`]: crate::traits::IndexedSeq::get
/// [`BitLength`]: crate::traits::BitLength
/// [`BitVecOps`]: crate::traits::BitVecOps
/// [`BitVecOpsMut`]: crate::traits::BitVecOpsMut
/// [`AtomicBitVecOps`]: crate::traits::AtomicBitVecOps
/// [`TryIntoUnaligned`]: crate::traits::TryIntoUnaligned
/// [`Unaligned`]: crate::traits::Unaligned
/// [`Word`]: crate::traits::Word
/// [`Backend`]: crate::traits::Backend
/// [lenders]: crate::utils::lenders
/// [`FromSlice`]: crate::utils::FromSlice
/// [`FromCloneableIntoIterator`]: crate::utils::FromCloneableIntoIterator
/// [`LineLender`]: crate::utils::LineLender
pub mod prelude {
    pub use crate::array::*;
    pub use crate::bal_paren::*;
    pub use crate::bits::*;
    pub use crate::dict::*;
    pub use crate::func::*;
    pub use crate::list::*;
    pub use crate::rank_sel::*;
    pub use crate::rank_small;
    pub use crate::traits::bit_field_slice;
    pub use crate::traits::bit_vec_ops;
    pub use crate::traits::indexed_dict;
    pub use crate::traits::{
        AtomicBitVecOps, Backend, BitLength, BitVecOps, BitVecOpsMut, TryIntoUnaligned, Unaligned,
        Word, iter::*, rank_sel::*,
    };
    pub use crate::utils::{FromCloneableIntoIterator, FromSlice, LineLender};
}

#[ambassador::delegatable_trait_remote]
pub(crate) trait Index<Idx> {
    type Output;
    fn index(&self, index: Idx) -> &Self::Output;
}

/// Parallel iterators performing very fast operations, such as [zeroing a bit
/// vector], should pass this argument to
/// [`IndexedParallelIterator::with_min_len`] or preferably
/// [`ParallelWithLen::with_len`].
///
/// [zeroing a bit vector]: crate::traits::BitVecOpsMut::reset
/// [`IndexedParallelIterator::with_min_len`]: rayon::iter::IndexedParallelIterator::with_min_len
/// [`ParallelWithLen::with_len`]: crate::ParallelWithLen::with_len
pub const RAYON_MIN_LEN: usize = 100_000;

/// Extension trait to fix Rayon's chunk size for parallel iterators.
///
/// This trait adds [`with_len`] to rayon's [`IndexedParallelIterator`], setting
/// both [`with_min_len`] and [`with_max_len`] to the same value to make chunks
/// approximately the specified size. Rayon splits work by recursive bisection,
/// so `len` acts as a split-size target: exact chunk boundaries and sizes are
/// not guaranteed (total work smaller than `len` is left unsplit).
///
/// This approach is useful for very fast operations of constant length per
/// element, like [zeroing a bit vector].
///
/// [zeroing a bit vector]: crate::traits::BitVecOpsMut::reset
/// [`with_len`]: ParallelWithLen::with_len
/// [`with_min_len`]: rayon::iter::IndexedParallelIterator::with_min_len
/// [`with_max_len`]: rayon::iter::IndexedParallelIterator::with_max_len
/// [`IndexedParallelIterator`]: rayon::iter::IndexedParallelIterator
#[cfg(feature = "rayon")]
pub trait ParallelWithLen: rayon::iter::IndexedParallelIterator + Sized {
    fn with_len(self, len: usize) -> rayon::iter::MaxLen<rayon::iter::MinLen<Self>> {
        use rayon::iter::IndexedParallelIterator;
        self.with_min_len(len).with_max_len(len)
    }
}

#[cfg(feature = "rayon")]
impl<T: rayon::iter::IndexedParallelIterator> ParallelWithLen for T {}

macro_rules! panic_if_out_of_bounds {
    ($index: expr, $len: expr) => {
        if $index >= $len {
            panic!("Index out of bounds: {} >= {}", $index, $len)
        }
    };
}

pub(crate) use panic_if_out_of_bounds;

macro_rules! panic_if_value {
    ($value: expr, $mask: expr, $bit_width: expr) => {
        if $value & $mask != $value {
            panic!("Value {} does not fit in {} bits", $value, $bit_width);
        }
    };
}
pub(crate) use panic_if_value;

macro_rules! debug_assert_bounds {
    ($index: expr, $len: expr) => {
        debug_assert!(
            $index < $len || ($index == 0 && $len == 0),
            "Index out of bounds: {} >= {}",
            $index,
            $len
        );
    };
}

pub(crate) use debug_assert_bounds;
