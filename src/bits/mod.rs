/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Structures for [bit vectors] and [vectors of bit fields of fixed width].
//!
//! These are the foundational storage types on which [rank/select
//! structures] and [indexed dictionaries] are built. The [`bit_vec!`] and
//! [`bit_field_vec!`] macros provide convenient construction.
//!
//! Bit vectors and bit-field vectors are parameterized by a [`Backend`], which
//! abstracts over different storage types (`Vec<W>`, `Box<[W]>`, memory-mapped
//! slices, etc.). They also implement the [`Backend`] trait themselves by
//! delegation.
//!
//! In particular, any [rank/select structure] requires a [`Backend`] and is a
//! [`Backend`] itself by delegation, so that every structure can be used as
//! the backend of a further structure.
//!
//! [`Backend`]: crate::traits::Backend
//! [bit vectors]: mod@bit_vec
//! [vectors of bit fields of fixed width]: mod@bit_field_vec
//! [rank/select structures]: crate::rank_sel
//! [rank/select structure]: crate::rank_sel
//! [indexed dictionaries]: crate::dict
//! [`bit_vec!`]: crate::bit_vec
//! [`bit_field_vec!`]: crate::bit_field_vec

pub mod bit_field_vec;
pub use bit_field_vec::*;

pub mod bit_vec;
pub use bit_vec::*;

pub use crate::bit_field_vec;
pub use crate::bit_vec;

// ── Unalign-read width predicates ───────────────────────────────────
//
// Three boolean predicates check whether a given bit width is safe
// for various flavors of unaligned read:
//
// * [`test_unaligned!`] — `$bw` is OK for [`BitFieldVec`]-style reads
//   at positions that are multiples of `$bw`.
// * [`test_unaligned_pos!`] — `$bw` is OK for a single
//   [`BitVec::get_value_unaligned_unchecked`] read at a specific
//   bit position `$pos`.
// * [`test_unaligned_any_pos!`] — `$bw` is OK for *every*
//   [`BitVec::get_value_unaligned_unchecked`] read regardless of
//   position (used when bulk-converting a structure to its unaligned
//   variant).
//
// The three differ in what they assume about the position. They
// return only a `bool`; call sites compose them with `assert!`,
// `debug_assert!`, or a `Result`-returning `if !test { return Err(…)
// }` block as appropriate. The error/panic message is the call
// site's responsibility — there's no point factoring it through a
// dedicated macro family because each call site gets exactly one of
// the three forms anyway.
//
// [`BitFieldVec`]: crate::bits::BitFieldVec
// [`BitVec::get_value_unaligned_unchecked`]: crate::bits::BitVec::get_value_unaligned_unchecked

/// Returns `true` if `$bw` satisfies the bit-width constraints for
/// *width-aligned* unaligned reads on word type `$ty` — that is, for
/// reads of `$bw` bits at positions that are multiples of `$bw`, as
/// used by [`BitFieldVec`](crate::bits::BitFieldVec).
///
/// If the type `$ty` has width *w*, the bit width `$bw` must be at
/// most *w* – 6, or exactly *w* – 4, or exactly *w*. These bounds are
/// the tightest that cover every `pos % 8` value attainable when `pos`
/// is a multiple of `$bw`.
macro_rules! test_unaligned {
    ($ty:ty, $bw:expr) => {{
        let bits = <$ty>::BITS as usize;
        $bw <= bits - 8 + 2 || $bw == bits - 8 + 4 || $bw == bits
    }};
}

pub(crate) use test_unaligned;

/// Returns `true` if `$bw` bits can be read at bit position `$pos`
/// using [`BitVec::get_value_unaligned_unchecked`] on word type `$ty`.
///
/// Position-dependent: `$bw + ($pos % 8) <= <$ty>::BITS`.
///
/// [`BitVec::get_value_unaligned_unchecked`]: crate::bits::BitVec::get_value_unaligned_unchecked
macro_rules! test_unaligned_pos {
    ($ty:ty, $pos:expr, $bw:expr) => {{
        let bits = <$ty>::BITS as usize;
        $bw + (($pos as usize) & 7) <= bits
    }};
}

pub(crate) use test_unaligned_pos;

/// Returns `true` if `$bw` bits can be read at *any* bit position
/// using [`BitVec::get_value_unaligned_unchecked`] on word type `$ty`.
///
/// Position-independent worst-case: `$bw + 7 <= <$ty>::BITS`.
///
/// [`BitVec::get_value_unaligned_unchecked`]: crate::bits::BitVec::get_value_unaligned_unchecked
macro_rules! test_unaligned_any_pos {
    ($ty:ty, $bw:expr) => {{
        let bits = <$ty>::BITS as usize;
        $bw + 7 <= bits
    }};
}

pub(crate) use test_unaligned_any_pos;
