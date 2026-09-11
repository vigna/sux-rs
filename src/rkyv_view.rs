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
//! This module bridges the gap with [`Lazy`], a plain reference into the
//! archive. The traits of this crate are implemented on it by converting, at
//! each call, just the fields the method reads, and delegating to the `Lazy`
//! view of the substructures; the structures of this crate are then built out
//! of `Lazy` backends, so no algorithm is duplicated and a benchmark comparing
//! the two formats measures the representation, not two implementations.
//!
//! The conversion is not cached: it is paid at every call, as it must be for a
//! format that is accessed in place. What a method does not pay for is the
//! pointers it does not read — for a structure as layered as
//! [`EliasFano`](crate::dict::EliasFano) those are most of them, as
//! [`get`](crate::traits::IndexedSeq::get) needs the inventory of the ones and
//! the low bits, and [`succ`](crate::traits::Succ) the inventory of the zeros
//! and the high bits.
//!
//! # Target Restrictions
//!
//! The views reinterpret archived word slices as native word slices, which is
//! correct only if archived words have the same size and byte order as native
//! ones. This holds when [rkyv] is compiled with the `pointer_width_64`
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

/// A lazy view of an archived structure.
///
/// Converting an archived structure *eagerly* — dereferencing every relative
/// pointer of the archive and rebuilding the whole structure — makes a query
/// pay for the substructures it never reads, which in a layered stack is most
/// of them.
///
/// `Lazy` defers the conversion. It is a plain reference into the archive, and
/// the traits of this crate are implemented on it by converting, at each call,
/// only the fields that the method reads, delegating to the `Lazy` view of the
/// substructures. Nothing is cached: the conversion is still paid at every
/// call, as it must be for a format that is accessed in place, but a method
/// pays just for the pointers it needs.
#[derive(Debug)]
#[repr(transparent)]
pub struct Lazy<'a, A: ?Sized>(pub &'a A);

impl<A: ?Sized> Clone for Lazy<'_, A> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<A: ?Sized> Copy for Lazy<'_, A> {}

/// Borrows from the archive the words underlying a [lazy view](Lazy).
///
/// [`AsRef`] cannot express this: it ties the returned slice to the borrow of
/// the view, whereas a view is just a reference into the archive, and the
/// words live as long as the archive does. Delegating from one view to the
/// view of a substructure therefore needs this trait, as the intermediate view
/// is a temporary.
pub trait Words<'a> {
    /// The type of the underlying words.
    type Word;

    /// Returns the underlying words, borrowed from the archive.
    fn words(self) -> &'a [Self::Word];
}
