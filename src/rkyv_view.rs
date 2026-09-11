/*
 *
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

//! Zero-copy access to [rkyv]-archived structures.
//!
//! [ε-serde] and [rkyv] solve the same problem in opposite ways. ε-serde
//! *substitutes generic parameters*: ε-copy deserializing an
//! [`EliasFano<u64, …, BitFieldVec<Box<[u64]>>>`](crate::dict::EliasFano)
//! yields the very same structure with `&[u64]` backends, so every method has
//! exactly the same code as the in-memory version. rkyv *mirrors types*: it
//! generates a parallel `ArchivedT` whose scalar fields are endianness-aware
//! wrappers and whose backends are reached through relative pointers, and that
//! type implements none of the traits of this crate.
//!
//! This module bridges the gap. Every archived structure of the Elias–Fano
//! stack gets a `to_native` method that dereferences the relative pointers and
//! converts the scalar fields, yielding the ordinary structure of this crate
//! with borrowed backends; the traits are then implemented on the archived
//! structure by delegation. No algorithm is duplicated, so a benchmark
//! comparing the two formats measures the representation, not two different
//! implementations.
//!
//! # Target Restrictions
//!
//! `to_native` reinterprets archived word slices as native word slices, which
//! is correct only if archived words have the same size and byte order as
//! native ones. This holds when [rkyv] is compiled with the `pointer_width_64`
//! feature (which this crate enables) and the target is little-endian and
//! 64-bit; the feature is not available on other targets.
//!
//! [ε-serde]: https://crates.io/crates/epserde
//! [rkyv]: https://crates.io/crates/rkyv

#[cfg(not(all(target_endian = "little", target_pointer_width = "64")))]
compile_error!(
    "the `rkyv` feature of `sux` requires a little-endian 64-bit target, as archived \
     word slices are reinterpreted as native word slices without conversion"
);

use crate::traits::Word;
use rkyv::primitive::{ArchivedU64, ArchivedUsize};

/// A word whose archived counterpart can be converted back into it.
///
/// This trait is implemented by `u64` and `usize`, whose archived counterpart
/// is [`ArchivedU64`] on the targets supported by this module.
pub trait ArchivedWord: Word + rkyv::Archive<Archived = ArchivedU64> {
    /// Converts an archived word into a native one.
    fn from_archived(archived: ArchivedU64) -> Self;
}

impl ArchivedWord for u64 {
    #[inline(always)]
    fn from_archived(archived: ArchivedU64) -> Self {
        archived.to_native()
    }
}

impl ArchivedWord for usize {
    #[inline(always)]
    fn from_archived(archived: ArchivedU64) -> Self {
        archived.to_native() as usize
    }
}

/// An archived structure that can be viewed as the corresponding structure of
/// this crate, borrowing its backends from the archive.
///
/// The conversion dereferences the relative pointers of the archive and
/// converts the scalar fields; it performs no allocation and copies no array.
/// The result implements all the traits of the original structure, with the
/// same code, so accessing it differs from accessing an in-memory structure
/// only by the cost of the conversion.
pub trait ToNative {
    /// The corresponding structure of this crate, borrowing from `self`.
    type Native<'a>
    where
        Self: 'a;

    /// Returns a view of this archived structure as the corresponding
    /// structure of this crate.
    fn to_native(&self) -> Self::Native<'_>;
}

/// Reinterprets a slice of archived words as a slice of native words.
///
/// On a little-endian 64-bit target [`ArchivedU64`] is `u64_le`, which is
/// `#[repr(transparent)]` over `u64` and therefore has the same layout as both
/// `u64` and `usize`.
///
/// # Safety
///
/// The archived words must have been written on a target with the same word
/// size and byte order, which the module-level target restrictions guarantee.
#[inline(always)]
pub(crate) unsafe fn native_words<W: ArchivedWord>(archived: &[ArchivedU64]) -> &[W] {
    debug_assert_eq!(size_of::<W>(), size_of::<ArchivedU64>());
    debug_assert_eq!(align_of::<W>(), align_of::<ArchivedU64>());
    // SAFETY: `W` is `u64` or `usize`, which on this target have the same
    // size, alignment and representation as `ArchivedU64`.
    unsafe { core::slice::from_raw_parts(archived.as_ptr().cast::<W>(), archived.len()) }
}

/// Converts an archived `usize` into a native one.
#[inline(always)]
pub(crate) fn native_usize(archived: ArchivedUsize) -> usize {
    archived.to_native() as usize
}
