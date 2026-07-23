/*
 *
 * SPDX-FileCopyrightText: 2024 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

use crate::traits::{IndexedSeq, IntoIteratorFrom, Types};

/// A newtype exhibiting a reference to a slice as an [indexed sequence].
///
/// Note that [`IndexedSeq`] is implemented for vectors, slices, and arrays,
/// so the need for this newtype is very limited.
///
/// You can create a [`SliceSeq`] with [`SliceSeq::new`], or with the equivalent
/// [`From`] implementation.
///
/// While a blanket implementation of [`IndexedSeq`] could be more convenient,
/// it would cause significant ambiguity problems.
///
/// [indexed sequence]: crate::traits::IndexedSeq
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SliceSeq<O: PartialEq + Copy, A: AsRef<[O]>>(A, std::marker::PhantomData<O>);

impl<O: PartialEq + Copy, A: AsRef<[O]>> SliceSeq<O, A> {
    #[must_use]
    pub fn new(slice: A) -> Self {
        Self(slice, std::marker::PhantomData)
    }
}

impl<O: PartialEq + Copy, A: AsRef<[O]>> From<A> for SliceSeq<O, A> {
    fn from(slice: A) -> Self {
        Self::new(slice)
    }
}

impl<O: PartialEq + Copy, A: AsRef<[O]>> Types for SliceSeq<O, A> {
    type Input = O;
    type Output<'a> = O;
}

impl<O: PartialEq + Copy, A: AsRef<[O]>> IndexedSeq for SliceSeq<O, A> {
    unsafe fn get_unchecked(&self, index: usize) -> Self::Output<'_> {
        // SAFETY: the caller guarantees index < self.len(), and self.len()
        // is exactly the length of this backing slice.
        unsafe { *self.0.as_ref().get_unchecked(index) }
    }

    fn get(&self, index: usize) -> Self::Output<'_> {
        // Snapshot the backing slice once. AsRef::as_ref gives no stability
        // guarantee across calls, so the default get (which bounds-checks
        // self.len() and then calls get_unchecked, each re-invoking
        // as_ref) could bounds-check one slice and index another. Binding the
        // slice once makes the checked access sound for any AsRef backend.
        let slice = self.0.as_ref();
        slice[index]
    }

    fn first_value(&self) -> Option<Self::Output<'_>> {
        self.0.as_ref().first().copied()
    }

    fn last_value(&self) -> Option<Self::Output<'_>> {
        self.0.as_ref().last().copied()
    }

    fn len(&self) -> usize {
        self.0.as_ref().len()
    }
}

impl<O: PartialEq + Copy, A: AsRef<[O]>> SliceSeq<O, A> {
    pub fn iter(&self) -> std::iter::Copied<std::slice::Iter<'_, O>> {
        self.0.as_ref().iter().copied()
    }
}

impl<'a, O: PartialEq + Copy, A: AsRef<[O]>> IntoIterator for &'a SliceSeq<O, A> {
    type Item = O;
    type IntoIter = std::iter::Copied<std::slice::Iter<'a, O>>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, O: PartialEq + Copy, A: AsRef<[O]>> IntoIteratorFrom for &'a SliceSeq<O, A> {
    type IntoIterFrom = std::iter::Skip<std::iter::Copied<std::slice::Iter<'a, O>>>;

    fn into_iter_from(self, from: usize) -> Self::IntoIterFrom {
        self.iter().skip(from)
    }
}
