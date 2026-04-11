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
/// unaligned reads on word type `$ty`.
///
/// If the type `$ty` has width *w*, the bit width `$bw` must
/// be at most *w* – 6, or exactly *w* – 4, or exactly *w*.
macro_rules! test_unaligned {
    ($ty:ty, $bw:expr) => {{
        let bits = <$ty>::BITS as usize;
        $bw <= bits - 8 + 2 || $bw == bits - 8 + 4 || $bw == bits
    }};
}

pub(crate) use test_unaligned;

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
