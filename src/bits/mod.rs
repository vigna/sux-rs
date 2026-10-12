/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

//! Structures for [bit vectors] and [vectors of bit fields of fixed width].
//!
//! These are the foundational storage types on which [rank/select
//! structures] and [indexed dictionaries] are built.
//!
//! | Type                         | Content                        | Concurrency                                 |
//! |------------------------------|--------------------------------|---------------------------------------------|
//! | [`BitVec`]                   | bits                           | none                                        |
//! | [`AtomicBitVec`]             | bits                           | atomic operations                           |
//! | [`BitFieldVec`]              | values of fixed bit width      | none                                        |
//! | [`AtomicBitFieldVec`]        | values of fixed bit width      | atomic operations (with [some caveats])     |
//! | `[W]`, `Vec<W>`, `[W; N]`    | values of bit width `W::BITS`  | atomic operations, if `W` is atomic         |
//!
//! The [`bit_vec!`] and [`bit_field_vec!`] macros provide convenient
//! construction, similarly to [`vec!`].
//!
//! Slices, vectors, and arrays of words are *full-width* bit-field slices:
//! they implement the same [bit-field slice traits] as [`BitFieldVec`] (or
//! [`AtomicBitFieldSlice`] for atomic words) with a bit width equal to that of
//! the word type, so code written for bit-field slices accepts them, too.
//!
//! Operations on bit vectors are provided by extension traits in
//! [`bit_vec_ops`], and operations on bit-field vectors by the traits in
//! [`bit_field_slice`], which must be imported explicitly (see the
//! documentation of each module for examples).
//!
//! # Backends
//!
//! Bit vectors and bit-field vectors are parameterized by a [`Backend`], which
//! abstracts over different storage types. A backend can be *owned*
//! (`Vec<W>`, which makes the structure growable, or `Box<[W]>`, which makes it
//! fixed-size) or *borrowed* (`&[W]` for read-only access, `&mut [W]` for
//! mutable access, and in particular memory-mapped or [ε-serde]-deserialized
//! slices). The default backend is `Vec<usize>` for [`BitVec`] and
//! [`BitFieldVec`], `Box<[Atomic<usize>]>` for [`AtomicBitVec`], and
//! `Vec<Atomic<usize>>` for [`AtomicBitFieldVec`]. All flavors can be converted
//! into one another, as explained in the conversion sections of the
//! [`bit_vec`](mod@bit_vec#conversions) and
//! [`bit_field_vec`](mod@bit_field_vec#conversions) modules.
//!
//! Bit vectors and bit-field vectors also implement the [`Backend`] trait
//! themselves by delegation. In particular, any [rank/select structure]
//! requires a [`Backend`] and is a [`Backend`] itself by delegation, so that
//! every structure can be used as the backend of a further structure.
//!
//! # Unaligned access
//!
//! Reading a value requires testing whether it spans two words, in which case
//! both words must be read. When the bit width makes it possible, a value can
//! be read instead with a branchless *unaligned* read of a word starting at
//! the byte containing its first bit, provided that there is a padding word at
//! the end of the backend, so that the read never exceeds the allocation.
//!
//! The trait [`TryIntoUnaligned`] converts a structure into a variant that
//! uses unaligned reads, adding a padding word if necessary; the [`Unaligned`]
//! type alias names the resulting type. A [`BitFieldVec`] with a `Box<[W]>`
//! backend converts into a [`BitFieldVecU`], and a [`BitVec`] with a
//! `Box<[W]>` backend into a [`BitVecU`], whose [`BitVecValueOps`] methods
//! use unaligned reads. Compound structures implement [`TryIntoUnaligned`] by
//! converting their components.
//!
//! The bit width must satisfy some constraints with respect to the bit width
//! `W::BITS` of the word type:
//!
//! - in a [`BitFieldVecU`], values start at multiples of the bit width, which
//!   must be at most `W::BITS` – 6, or exactly `W::BITS` – 4, or exactly
//!   `W::BITS`, or the conversion fails;
//! - in a [`BitVecU`], values start at arbitrary positions, so reading `width`
//!   bits at position `pos` is possible only if `width + pos % 8 ≤ W::BITS`:
//!   only widths up to `W::BITS` – 7 are always available, and reads
//!   violating the constraint panic.
//!
//! [`Backend`]: crate::traits::Backend
//! [bit vectors]: mod@bit_vec
//! [vectors of bit fields of fixed width]: mod@bit_field_vec
//! [rank/select structures]: crate::rank_sel
//! [rank/select structure]: crate::rank_sel
//! [indexed dictionaries]: crate::dict
//! [`bit_vec!`]: crate::bit_vec
//! [`bit_field_vec!`]: crate::bit_field_vec
//! [some caveats]: AtomicBitFieldVec
//! [bit-field slice traits]: crate::traits::bit_field_slice
//! [`AtomicBitFieldSlice`]: crate::traits::AtomicBitFieldSlice
//! [`bit_vec_ops`]: crate::traits::bit_vec_ops
//! [`bit_field_slice`]: crate::traits::bit_field_slice
//! [ε-serde]: https://crates.io/crates/epserde
//! [`TryIntoUnaligned`]: crate::traits::TryIntoUnaligned
//! [`Unaligned`]: crate::traits::Unaligned
//! [`BitVecValueOps`]: crate::traits::BitVecValueOps

pub mod bit_field_vec;
pub use bit_field_vec::*;

pub mod bit_vec;
pub use bit_vec::*;

pub use crate::bit_field_vec;
pub use crate::bit_vec;

// Explicit re-exports of the main types, so that they are listed in the
// documentation of this module (inlining the glob re-exports would list also
// hidden and auxiliary items).
#[doc(inline)]
pub use bit_field_vec::{AtomicBitFieldVec, BitFieldVec, BitFieldVecU};
#[doc(inline)]
pub use bit_vec::{AtomicBitVec, BitVec, BitVecU};

/// Implements the conversions between the flavors of a vector-like structure
/// `$Vec` and of its atomic counterpart `$AtomicVec`.
///
/// Both structures must have a backend field `bits` and the same additional
/// fields `$field`. The lists of generic parameters (with bounds) are for
/// borrowing conversions (the parameter `B` of `$Vec` and `$AtomicVec`), for
/// conversions between `Vec` and `Box` backends of the same kind (the
/// parameter `W` of `$Vec` and `A` of `$AtomicVec`), and for conversions
/// between atomic and non-atomic backends (the parameter `W`).
macro_rules! impl_conversions {
    (
        $Vec:ident, $AtomicVec:ident { $($field:ident),* },
        borrow: [$($bb:tt)*], [$($abb:tt)*],
        owned: [$($wb:tt)*], [$($ab:tt)*],
        atomic: [$($awb:tt)*] $(,)?
    ) => {
        // Borrowing conversions

        impl<'a, $($bb)* + AsRef<[B::Word]>> From<&'a $Vec<B>> for $Vec<&'a [B::Word]> {
            fn from(value: &'a $Vec<B>) -> Self {
                $Vec {
                    bits: value.bits.as_ref(),
                    $($field: value.$field,)*
                }
            }
        }

        impl<'a, $($abb)* + AsRef<[B::Word]>> From<&'a $AtomicVec<B>>
            for $AtomicVec<&'a [B::Word]>
        {
            fn from(value: &'a $AtomicVec<B>) -> Self {
                $AtomicVec {
                    bits: value.bits.as_ref(),
                    $($field: value.$field,)*
                }
            }
        }

        impl<'a, $($bb)* + AsMut<[B::Word]>> From<&'a mut $Vec<B>>
            for $Vec<&'a mut [B::Word]>
        {
            fn from(value: &'a mut $Vec<B>) -> Self {
                $Vec {
                    bits: value.bits.as_mut(),
                    $($field: value.$field,)*
                }
            }
        }

        impl<'a, $($abb)* + AsMut<[B::Word]>> From<&'a mut $AtomicVec<B>>
            for $AtomicVec<&'a mut [B::Word]>
        {
            fn from(value: &'a mut $AtomicVec<B>) -> Self {
                $AtomicVec {
                    bits: value.bits.as_mut(),
                    $($field: value.$field,)*
                }
            }
        }

        /// This conversion may fail if the alignment of `W::Atomic` is stricter
        /// than that of `W`.
        impl<'a, $($awb)*> TryFrom<$Vec<&'a mut [W]>> for $AtomicVec<&'a mut [W::Atomic]> {
            type Error = $crate::utils::InsufficientAlignmentError;

            #[inline]
            fn try_from(value: $Vec<&'a mut [W]>) -> ::core::result::Result<Self, Self::Error> {
                if core::mem::align_of::<W::Atomic>() > core::mem::align_of::<W>() {
                    return Err($crate::utils::InsufficientAlignmentError::new::<W, W::Atomic>());
                }
                Ok($AtomicVec {
                    // SAFETY: an atomic type has the same size and in-memory
                    // representation as its value type, and the alignment of the
                    // target type is not stricter than that of the source type (just
                    // checked), which the buffer is known to satisfy. No deallocation
                    // happens through this reference, so the alignments need not be
                    // equal.
                    bits: unsafe { core::mem::transmute::<&'a mut [W], &'a mut [W::Atomic]>(value.bits) },
                    $($field: value.$field,)*
                })
            }
        }

        /// This conversion is infallible, contrarily to the [opposite one], because
        /// the alignment of an atomic type is never less strict than the alignment of
        /// its value type.
        ///
        #[doc = concat!(
            "[opposite one]: #impl-TryFrom%3C", stringify!($Vec),
            "%3C%26mut+%5BW%5D%3E%3E-for-", stringify!($AtomicVec),
            "%3C%26mut+%5B%3CW+as+AtomicPrimitive%3E::Atomic%5D%3E"
        )]
        impl<'a, $($awb)*> From<$AtomicVec<&'a mut [W::Atomic]>> for $Vec<&'a mut [W]> {
            #[inline]
            fn from(value: $AtomicVec<&'a mut [W::Atomic]>) -> Self {
                $Vec {
                    // SAFETY: an atomic type has the same size and in-memory
                    // representation as its value type. Moreover, all atomic types
                    // have an alignment at least as strict as that of their value
                    // type, so the buffer, which is aligned for W::Atomic, is aligned
                    // for W, too. No deallocation happens through this reference, so
                    // the alignments need not be equal.
                    bits: unsafe { core::mem::transmute::<&'a mut [W::Atomic], &'a mut [W]>(value.bits) },
                    $($field: value.$field,)*
                }
            }
        }

        // Conversions between owned backends of the same kind

        impl<$($wb)*> From<$Vec<Vec<W>>> for $Vec<Box<[W]>> {
            fn from(value: $Vec<Vec<W>>) -> Self {
                $Vec {
                    bits: value.bits.into_boxed_slice(),
                    $($field: value.$field,)*
                }
            }
        }

        impl<$($wb)*> From<$Vec<Box<[W]>>> for $Vec<Vec<W>> {
            fn from(value: $Vec<Box<[W]>>) -> Self {
                $Vec {
                    bits: value.bits.into_vec(),
                    $($field: value.$field,)*
                }
            }
        }

        impl<$($ab)*> From<$AtomicVec<Vec<A>>> for $AtomicVec<Box<[A]>> {
            fn from(value: $AtomicVec<Vec<A>>) -> Self {
                $AtomicVec {
                    bits: value.bits.into_boxed_slice(),
                    $($field: value.$field,)*
                }
            }
        }

        impl<$($ab)*> From<$AtomicVec<Box<[A]>>> for $AtomicVec<Vec<A>> {
            fn from(value: $AtomicVec<Box<[A]>>) -> Self {
                $AtomicVec {
                    bits: value.bits.into_vec(),
                    $($field: value.$field,)*
                }
            }
        }

        // Conversions between atomic and non-atomic owned backends

        $crate::bits::impl_conversions!(@try_from [$($awb)*]
            $Vec[Vec<W>] => $AtomicVec[Vec<W::Atomic>]
            via $crate::utils::transmute_vec_into_atomic; $($field),*);
        $crate::bits::impl_conversions!(@try_from [$($awb)*]
            $Vec[Vec<W>] => $AtomicVec[Box<[W::Atomic]>]
            via $crate::utils::transmute_vec_into_atomic; $($field),*);
        $crate::bits::impl_conversions!(@try_from [$($awb)*]
            $Vec[Box<[W]>] => $AtomicVec[Vec<W::Atomic>]
            via $crate::utils::transmute_boxed_slice_into_atomic; $($field),*);
        $crate::bits::impl_conversions!(@try_from [$($awb)*]
            $Vec[Box<[W]>] => $AtomicVec[Box<[W::Atomic]>]
            via $crate::utils::transmute_boxed_slice_into_atomic; $($field),*);
        $crate::bits::impl_conversions!(@try_from [$($awb)*]
            $AtomicVec[Vec<W::Atomic>] => $Vec[Vec<W>]
            via $crate::utils::transmute_vec_from_atomic; $($field),*);
        $crate::bits::impl_conversions!(@try_from [$($awb)*]
            $AtomicVec[Vec<W::Atomic>] => $Vec[Box<[W]>]
            via $crate::utils::transmute_vec_from_atomic; $($field),*);
        $crate::bits::impl_conversions!(@try_from [$($awb)*]
            $AtomicVec[Box<[W::Atomic]>] => $Vec[Vec<W>]
            via $crate::utils::transmute_boxed_slice_from_atomic; $($field),*);
        $crate::bits::impl_conversions!(@try_from [$($awb)*]
            $AtomicVec[Box<[W::Atomic]>] => $Vec[Box<[W]>]
            via $crate::utils::transmute_boxed_slice_from_atomic; $($field),*);
    };

    (@try_from [$($g:tt)*] $Src:ident[$SB:ty] => $Dst:ident[$DB:ty]
        via $transmute:path; $($field:ident),*) => {
        /// This conversion may fail if the alignment of `W` is not the same as
        /// that of `W::Atomic`; in that case, the error contains the original
        /// vector.
        impl<$($g)*> TryFrom<$Src<$SB>> for $Dst<$DB> {
            type Error = $crate::utils::DifferentAlignmentError<$Src<$SB>>;

            #[inline]
            fn try_from(value: $Src<$SB>) -> ::core::result::Result<Self, Self::Error> {
                let $Src { bits, $($field),* } = value;
                match $transmute(bits) {
                    // From::from converts between Vec and Box, if necessary
                    Ok(bits) => Ok($Dst {
                        bits: From::from(bits),
                        $($field,)*
                    }),
                    Err(error) => Err($crate::utils::DifferentAlignmentError(
                        error.0,
                        $Src {
                            bits: error.1,
                            $($field,)*
                        },
                    )),
                }
            }
        }
    };
}

pub(crate) use impl_conversions;

// ── Unalign-read width predicates ───────────────────────────────────
//
// Three boolean predicates check whether a given bit width is safe
// for various flavors of unaligned read:
//
// * [test_unaligned!]: $bw is OK for [BitFieldVec]-style reads
//   at positions that are multiples of $bw.
// * [test_unaligned_pos!]: $bw is OK for a single
//   [BitVec::get_value_unaligned_unchecked] read at a specific
//   bit position $pos.
// * [test_unaligned_any_pos!]: $bw is OK for *every*
//   [BitVec::get_value_unaligned_unchecked] read regardless of
//   position (used when bulk-converting a structure to its unaligned
//   variant).
//
// The three differ in what they assume about the position. They
// return only a bool; call sites compose them with assert!,
// debug_assert!, or a Result-returning if !test { return Err(…)
// } block as appropriate. The error/panic message is the call
// site's responsibility; there's no point factoring it through a
// dedicated macro family because each call site gets exactly one of
// the three forms anyway.
//
// [BitFieldVec]: crate::bits::BitFieldVec
// [BitVec::get_value_unaligned_unchecked]: crate::traits::BitVecOps::get_value_unaligned_unchecked

/// Returns `true` if `$bw` satisfies the bit-width constraints for
/// *width-aligned* unaligned reads on word type `$ty`, that is, for
/// reads of `$bw` bits at positions that are multiples of `$bw`, as
/// used by [`BitFieldVec`].
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
/// [`BitVec::get_value_unaligned_unchecked`]: crate::traits::BitVecOps::get_value_unaligned_unchecked
macro_rules! test_unaligned_pos {
    ($ty:ty, $pos:expr, $bw:expr) => {{
        let bits = <$ty>::BITS as usize;
        let bw = $bw;
        bw <= bits && (($pos as usize) & 7) <= bits - bw
    }};
}

pub(crate) use test_unaligned_pos;

/// Returns `true` if `$bw` bits can be read at *any* bit position
/// using [`BitVec::get_value_unaligned_unchecked`] on word type `$ty`.
///
/// Position-independent worst-case: `$bw + 7 <= <$ty>::BITS`.
///
/// [`BitVec::get_value_unaligned_unchecked`]: crate::traits::BitVecOps::get_value_unaligned_unchecked
macro_rules! test_unaligned_any_pos {
    ($ty:ty, $bw:expr) => {{
        let bits = <$ty>::BITS as usize;
        let bw = $bw;
        bw <= bits - 7
    }};
}

pub(crate) use test_unaligned_any_pos;
