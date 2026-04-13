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

/// Returns `true` if `$bw` satisfies the bit-width constraints for
/// *width-aligned* unaligned reads on word type `$ty` — that is, for
/// reads of `$bw` bits at positions that are multiples of `$bw`, as
/// used by [`BitFieldVec`](crate::bits::BitFieldVec).
///
/// If the type `$ty` has width *w*, the bit width `$bw` must be at
/// most *w* – 6, or exactly *w* – 4, or exactly *w*. These bounds are
/// the tightest that cover every `pos % 8` value attainable when `pos`
/// is a multiple of `$bw`.
///
/// # Which check to use
///
/// * For reads at positions that are multiples of the value width
///   ([`BitFieldVec`](crate::bits::BitFieldVec)), use
///   this test.
/// * For a single read at a *specific* arbitrary position, use
///   [`test_unaligned_pos!`].
/// * For bulk eligibility, that is, "can every width-`$bw` read succeed
///   regardless of where it lands?", use [`test_unaligned_any_pos!`].
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
/// The check is *position-dependent*: a single read of width `$bw` at
/// position `$pos` succeeds iff `$bw + ($pos % 8) <= <$ty>::BITS`.
/// Use this when you know both the position and the width of a
/// specific read.
///
/// For a position-independent check, that is, "can every read of this width
/// succeed regardless of where it lands?", use [`test_unaligned_any_pos!`].
///
/// This differs from [`test_unaligned!`], which models
/// [`BitFieldVec`](crate::bits::BitFieldVec)-style width-aligned
/// positions and can therefore allow looser widths such as `W::BITS`.
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
/// The bound is `$bw <= <$ty>::BITS - 7`, covering the worst case
/// `pos % 8 == 7`. Use this when you need to guarantee that every
/// read of a given width succeeds regardless of where it lands —
/// that is, when deciding whether a compound structure can be bulk-
/// converted to its unaligned variant.
///
/// For a position-dependent check against a specific position, use
/// [`test_unaligned_pos!`]. For `BitFieldVec`-style width-aligned
/// positions, use [`test_unaligned!`].
///
/// [`BitVec::get_value_unaligned_unchecked`]: crate::bits::BitVec::get_value_unaligned_unchecked
macro_rules! test_unaligned_any_pos {
    ($ty:ty, $bw:expr) => {{
        let bits = <$ty>::BITS as usize;
        $bw + 7 <= bits
    }};
}

pub(crate) use test_unaligned_any_pos;

/// Returns an error if `$bw` does not satisfy the bit-width constraints for
/// unaligned reads on word type `$ty`.
///
/// Uses the same constraints as [`test_unaligned!`].
macro_rules! ensure_unaligned {
    ($ty:ty, $bw:expr) => {
        if !test_unaligned!($ty, $bw) {
            let bits = <$ty>::BITS as usize;
            return Err($crate::traits::UnalignedConversionError(format!(
                "bit width {} does not satisfy the constraints for unaligned reads on word type {} (must be <= {}, or == {}, or == {})",
                $bw, stringify!($ty), bits - 8 + 2, bits - 8 + 4, bits,
            )));
        }
    };
}

pub(crate) use ensure_unaligned;

/// Returns an error if `$bw` does not satisfy the bit-width constraints
/// for arbitrary-position unaligned reads on word type `$ty`.
///
/// Uses the same constraints as [`test_unaligned_any_pos!`].
macro_rules! ensure_unaligned_any_pos {
    ($ty:ty, $bw:expr) => {
        if !test_unaligned_any_pos!($ty, $bw) {
            return Err($crate::traits::UnalignedConversionError(format!(
                "bit width {} does not satisfy the constraints for arbitrary-position unaligned reads on word type {} (must be <= {})",
                $bw, stringify!($ty), <$ty>::BITS as usize - 7,
            )));
        }
    };
}

pub(crate) use ensure_unaligned_any_pos;

/// Panics if `$bw` does not satisfy the bit-width constraints for unaligned reads
/// on word type `$ty`.
///
/// Uses the same constraints as [`test_unaligned!`].
macro_rules! assert_unaligned {
    ($ty:ty, $bw:expr) => {
        assert!(test_unaligned!($ty, $bw),
            "bit width {} does not satisfy the constraints for unaligned reads on word type {} (must be <= {}, or == {}, or == {})",
            $bw, stringify!($ty), <$ty>::BITS as usize - 8 + 2, <$ty>::BITS as usize - 8 + 4, <$ty>::BITS as usize,
        );
    };
}

pub(crate) use assert_unaligned;

/// Panics if the read of `$bw` bits at bit position `$pos` does not
/// satisfy the position-dependent unaligned-read constraint on word
/// type `$ty`.
///
/// Uses the same constraint as [`test_unaligned_pos!`]:
/// `$bw + ($pos % 8) <= <$ty>::BITS`.
macro_rules! assert_unaligned_pos {
    ($ty:ty, $pos:expr, $bw:expr) => {
        assert!(test_unaligned_pos!($ty, $pos, $bw),
            "bit width {} at bit position {} does not fit in a single unaligned read on word type {} (width + (pos % 8) must be <= {})",
            $bw, $pos, stringify!($ty), <$ty>::BITS as usize,
        );
    };
}

pub(crate) use assert_unaligned_pos;

/// Panics in debug mode if `$bw` does not satisfy the bit-width constraints for
/// unaligned reads on word type `$ty`.
///
/// Uses the same constraints as [`test_unaligned!`].
macro_rules! debug_assert_unaligned {
    ($ty:ty, $bw:expr) => {
        debug_assert!(test_unaligned!($ty, $bw),
            "bit width {} does not satisfy the constraints for unaligned reads on word type {} (must be <= {}, or == {}, or == {})",
            $bw, stringify!($ty), <$ty>::BITS as usize - 8 + 2, <$ty>::BITS as usize - 8 + 4, <$ty>::BITS as usize,
        );
    };
}

pub(crate) use debug_assert_unaligned;

/// Panics in debug mode if the read of `$bw` bits at bit position
/// `$pos` does not satisfy the position-dependent unaligned-read
/// constraint on word type `$ty`.
///
/// Uses the same constraint as [`test_unaligned_pos!`].
macro_rules! debug_assert_unaligned_pos {
    ($ty:ty, $pos:expr, $bw:expr) => {
        debug_assert!(test_unaligned_pos!($ty, $pos, $bw),
            "bit width {} at bit position {} does not fit in a single unaligned read on word type {} (width + (pos % 8) must be <= {})",
            $bw, $pos, stringify!($ty), <$ty>::BITS as usize,
        );
    };
}

pub(crate) use debug_assert_unaligned_pos;
