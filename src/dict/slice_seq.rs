/*
 *
 * SPDX-FileCopyrightText: 2024 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Adapters from reference to slices to [indexed sequences](crate::traits::IndexedSeq).

use crate::traits::{IndexedSeq, IntoIteratorFrom, Types};

/// A newtype exhibiting a reference to a slice as an [indexed
/// sequence](crate::traits::IndexedSeq).
///
/// You can create a [`SliceSeq`] with [`SliceSeq::new`], or with the equivalent
/// [`From`] implementation.
///
/// While a blanket implementation of [`IndexedSeq`] could be more convenient,
/// it would cause significant ambiguity problems.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SliceSeq<O: PartialEq<usize> + PartialEq + Copy, A: AsRef<[O]>>(
    A,
    std::marker::PhantomData<O>,
)
where
    usize: PartialEq<O>;

impl<O: PartialEq<usize> + PartialEq + Copy, A: AsRef<[O]>> SliceSeq<O, A>
where
    usize: PartialEq<O>,
{
    pub fn new(slice: A) -> Self {
        Self(slice, std::marker::PhantomData)
    }
}

impl<O: PartialEq<usize> + PartialEq + Copy, A: AsRef<[O]>> From<A> for SliceSeq<O, A>
where
    usize: PartialEq<O>,
{
    fn from(slice: A) -> Self {
        Self::new(slice)
    }
}

impl<O: PartialEq<usize> + PartialEq + Copy, A: AsRef<[O]>> Types for SliceSeq<O, A>
where
    usize: PartialEq<O>,
{
    type Input = usize;
    type Output = O;
}

impl<O: PartialEq<usize> + PartialEq + Copy, A: AsRef<[O]>> IndexedSeq for SliceSeq<O, A>
where
    usize: PartialEq<O>,
{
    unsafe fn get_unchecked(&self, index: usize) -> Self::Output {
        unsafe { *self.0.as_ref().get_unchecked(index) }
    }

    fn len(&self) -> usize {
        self.0.as_ref().len()
    }
}

impl<O: PartialEq<usize> + PartialEq + Copy, A: AsRef<[O]>> SliceSeq<O, A>
where
    usize: PartialEq<O>,
{
    pub fn iter(&self) -> std::iter::Copied<std::slice::Iter<'_, O>> {
        self.0.as_ref().iter().copied()
    }
}

impl<'a, O: PartialEq<usize> + PartialEq + Copy, A: AsRef<[O]>> IntoIterator for &'a SliceSeq<O, A>
where
    usize: PartialEq<O>,
{
    type Item = O;
    type IntoIter = std::iter::Copied<std::slice::Iter<'a, O>>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, O: PartialEq<usize> + PartialEq + Copy, A: AsRef<[O]>> IntoIteratorFrom
    for &'a SliceSeq<O, A>
where
    usize: PartialEq<O>,
{
    type IntoIterFrom = std::iter::Skip<std::iter::Copied<std::slice::Iter<'a, O>>>;

    fn into_iter_from(self, from: usize) -> Self::IntoIterFrom {
        self.iter().skip(from)
    }
}
