/*
 *
 * SPDX-FileCopyrightText: 2024 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Adapters from reference to slices to [indexed sequences](crate::traits::IndexedSeq).

use crate::traits::{IndexedSeq, Types};

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

    fn iter(&self) -> impl Iterator<Item = Self::Output> {
        self.0.as_ref().iter().copied()
    }
}
