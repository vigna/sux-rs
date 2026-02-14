/*
 *
 * SPDX-FileCopyrightText: 2023 Tommaso Fontana
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! An implementation of the Elias–Fano representation of monotone sequences.
//!
//! Given a monotone sequence 0 ≤ *x*₀ ≤ *x*₁ ≤ ... ≤ *x*<sub>*n* – 1</sub> ≤
//! *u*, where *u* is a given upper bound, the Elias–Fano representation makes
//! it possible to store the sequence using at most 2 + lg(*u*/*n*) bits per
//! element, which is very close to the information-theoretical lower bound ≈ lg
//! *e* + lg(*u*/*n*) when *n* is much smaller than *u*. A typical example is a
//! list of pointer into records of a large file: instead of using, for each
//! pointer, a number of bit sufficient to express the length of the file, the
//! Elias–Fano representation makes it possible to use, for each pointer, a
//! number of bits roughly equal to the logarithm of the average length of a
//! record.
//!
//! The representation was introduced in Peter Elias in “[Efficient storage and
//! retrieval by content and address of static
//! files](https://dl.acm.org/doi/abs/10.1145/321812.321820)”, *J. Assoc.
//! Comput. Mach.*, 21(2):246–260, ACM, 1974, and also independently by Robert
//! Fano in “[On the number of bits required to implement an associative
//! memory](http://csg.csail.mit.edu/pubs/memos/Memo-61/Memo-61.pdf)”,
//! Memorandum 61, Computer Structures Group, Project MAC, MIT, Cambridge,
//! Mass., n.d., 1971.
//!
//! This implementation is based on algorithmic engineering ideas proposed by
//! Sebastiano Vigna in “[Quasi-succinct
//! indices](https://dl.acm.org/doi/10.1145/2433396.2433409)”, *Proceedings of
//! the 6th ACM International Conference on Web Search and Data Mining,
//! WSDM'13*, pages 83–92, ACM, 2013. The name “Elias–Fano” for this
//! representation was used for the first time by  Sebastiano Vigna in
//! “[Broadword Implementation of Rank/Select
//! Queries](https://link.springer.com/chapter/10.1007/978-3-540-68552-4_12)”,
//! _Proc. of the 7th International Workshop on Experimental Algorithms, WEA
//! 2008_, volume 5038 of Lecture Notes in Computer Science, pages 154–168,
//! Springer, 2008.
//!
//! The elements of the sequence are recorded by storing separately the lower
//! *s* = ⌊lg(*u*/*n*)⌋ bits and the remaining upper bits. The lower bits are
//! stored contiguously, whereas the upper bits are stored in an array of *n* +
//! ⌊*u* / 2<sup>*s*</sup>⌋ bits by setting, for each 0 ≤ *i* < *n*, the bit of
//! index ⌊*x*<sub>*i*</sub> / 2<sup>*s*</sup>⌋ + *i*; the value can then be
//! recovered by selecting the *i*-th bit of the resulting bit array and
//! subtracting *i* (note that this will work because the upper bits are
//! nondecreasing).
//!
//!

use crate::prelude::{indexed_dict::*, *};
use crate::traits::{AtomicBitVecOps, BitVecOpsMut, bit_field_slice::*};
use common_traits::SelectInWord;
use core::sync::atomic::Ordering;
use mem_dbg::*;
use std::borrow::Borrow;
use std::iter::FusedIterator;
use value_traits::slices::{SliceByValue, SliceByValueMut};

/// The default type for an Elias–Fano structure implementing an [`IndexedSeq`].
///
/// You can start from this type to customize your Elias–Fano structure using
/// different const parameters or a different selection structure altogether.
pub type EfSeq = EliasFano<SelectAdaptConst<BitVec<Box<[usize]>>, Box<[usize]>, 12, 3>>;

/// The default type for an Elias–Fano structure implementing an
/// [`SuccUnchecked`] and [`PredUnchecked`].
///
/// You can start from this type to customize your Elias–Fano structure using
/// different const parameters or a different selection structure altogether.
pub type EfDict = EliasFano<SelectZeroAdaptConst<BitVec<Box<[usize]>>, Box<[usize]>, 12, 3>>;

/// The default type for an Elias–Fano structure implementing an
/// [`IndexedDict`], [`Succ`], and [`Pred`].
///
/// You can start from this type to customize your Elias–Fano structure using
/// different const parameters or different selection structures altogether.
pub type EfSeqDict = EliasFano<
    SelectZeroAdaptConst<
        SelectAdaptConst<BitVec<Box<[usize]>>, Box<[usize]>, 12, 3>,
        Box<[usize]>,
        12,
        3,
    >,
>;

/// An [`IndexedDict`] that stores a monotone sequence of integers using the
/// Elias–Fano representation.
///
/// There are two main ways to build a base [`EliasFano`] structure: creating an
/// [`EliasFanoBuilder`], or an [`EliasFanoConcurrentBuilder`], and then adding
/// values using `push` or `extend`. Additionally, a [`From`] convenience
/// implementation makes it possible to build an [`EliasFano`] from a slice.
///
/// In both cases, if you use the [`build`](EliasFanoBuilder::build) method you
/// will only be able to iterate over the sequence. Using the methods
/// [`build_with_seq`](EliasFanoBuilder::build_with_seq),
/// [`build_with_dict`](EliasFanoBuilder::build_with_dict), or
/// [`build_with_seq_and_dict`](EliasFanoBuilder::build_with_seq_and_dict) you
/// will have access to the additional functionalities of an [`IndexedSeq`] or
/// an [`IndexedDict`] with [`SuccUnchecked`] and [`PredUnchecked`], or both
/// (and in that case, [`Succ`] and [`Pred`]).
///
/// It is also possible to enrich manually the base structure by calling
/// [`EliasFano::map_high_bits`]. To use the structure as an [`IndexedSeq`] you
/// need to add a selection structure for ones, whereas to use it as an
/// [`IndexedDict`] with [`SuccUnchecked`] and [`PredUnchecked`] you need to add
/// a selection structure for zeros. [`SelectAdaptConst`] and
/// [`SelectZeroAdaptConst`] are the structures of choice for this purpose. If
/// you add both structures, you will have an [`IndexedDict`] with [`Succ`] and
/// [`Pred`].
///
/// # Examples
///
/// Using convenience builders:
/// ```rust
/// # use sux::rank_sel::{SelectAdaptConst, SelectZeroAdaptConst};
/// # use sux::dict::{EliasFanoBuilder};
/// # use sux::traits::{Types,IndexedSeq,IndexedDict,SuccUnchecked,Succ};
/// let mut efb = EliasFanoBuilder::new(4, 10);
/// efb.push(0);
/// efb.push(2);
/// efb.push(8);
/// efb.push(10);
///
/// let ef = efb.build_with_seq();
///
/// assert_eq!(ef.get(0), 0);
/// assert_eq!(ef.get(1), 2);
///
/// let mut efb = EliasFanoBuilder::new(4, 10);
/// efb.push(0);
/// efb.push(2);
/// efb.push(8);
/// efb.push(10);
///
/// let ef = efb.build_with_dict();
///
/// assert_eq!(unsafe { ef.succ_unchecked::<false>(6) }, (2, 8));
/// // This would panic: ef.succ_unchecked(11)
///
/// let mut efb = EliasFanoBuilder::new(4, 10);
/// efb.push(0);
/// efb.push(2);
/// efb.push(8);
/// efb.push(10);
///
/// let ef = efb.build_with_seq_and_dict();
/// assert_eq!(ef.get(0), 0);
/// assert_eq!(ef.get(1), 2);
/// assert_eq!(ef.succ(6), Some((2, 8)));
/// assert_eq!(ef.succ(11), None);
/// ```
///
/// Enriching manually a base structure with
/// [`map_high_bits`](EliasFano::map_high_bits):
/// ```rust
/// # use sux::rank_sel::{SelectAdaptConst, SelectZeroAdaptConst};
/// # use sux::dict::{EliasFanoBuilder};
/// # use sux::traits::{Types,IndexedSeq,IndexedDict,Succ};
/// let mut efb = EliasFanoBuilder::new(4, 10);
/// efb.push(0);
/// efb.push(2);
/// efb.push(8);
/// efb.push(10);
///
/// let ef = efb.build();
/// // Add a selection structure for ones (implements IndexedSeq)
/// let ef = unsafe { ef.map_high_bits(SelectAdaptConst::<_, _>::new) };
///
/// assert_eq!(ef.get(0), 0);
/// assert_eq!(ef.get(1), 2);
///
/// // Add a further selection structure for zeros (implements IndexedDict, Succ, Pred)
/// let ef = unsafe { ef.map_high_bits(SelectZeroAdaptConst::<_, _>::new) };
///
/// assert_eq!(ef.succ(6), Some((2, 8)));
/// assert_eq!(ef.succ(11), None);
/// ```
///
/// Building a base structure with convenience methods:
/// ```rust
/// # use sux::rank_sel::{SelectAdaptConst};
/// # use sux::dict::{EliasFano, EliasFanoBuilder};
/// # use sux::traits::{Types,IndexedSeq};
///
/// // Convenience constructor that iterates over a slice
/// let mut ef: EliasFano = vec![0, 2, 8, 10].into();
/// // Add a selection structure for ones (implements IndexedSeq)
/// let ef = unsafe { ef.map_high_bits(SelectAdaptConst::<_, _>::new) };
///
/// assert_eq!(ef.get(0), 0);
/// assert_eq!(ef.get(1), 2);
///
/// let mut efb = EliasFanoBuilder::new(4, 10);
/// // Add values using an iterator
/// efb.extend(vec![0, 2, 8, 10]);
/// let ef = efb.build();
/// // Add a selection structure for ones (implements IndexedSeq)
/// let ef = unsafe { ef.map_high_bits(SelectAdaptConst::<_, _>::new) };
///
/// assert_eq!(ef.get(0), 0);
/// assert_eq!(ef.get(1), 2);
/// ```

#[derive(Debug, Clone, Copy, Hash, MemDbg, MemSize, value_traits::Subslices)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[value_traits_subslices(bound = "H: AsRef<[usize]> + SelectUnchecked")]
#[value_traits_subslices(bound = "L: SliceByValue<Value = usize>")]
pub struct EliasFano<H = BitVec<Box<[usize]>>, L = BitFieldVec<usize, Box<[usize]>>> {
    /// The number of values.
    n: usize,
    /// An upper bound to the values.
    u: usize,
    /// The number of lower bits.
    l: usize,
    /// The lower-bits array.
    low_bits: L,
    /// the higher-bits array.
    high_bits: H,
}

impl<H, L> EliasFano<H, L> {
    /// Returns the parts composing the structure (number of elements, upper
    /// bound, number of lower bits, low bits, high bits).
    pub fn into_parts(self) -> (usize, usize, usize, L, H) {
        (self.n, self.u, self.l, self.low_bits, self.high_bits)
    }

    /// Estimate the size of an instance.
    pub fn estimate_size(u: usize, n: usize) -> usize {
        2 * n + (n * (u as f64 / n as f64).log2().ceil() as usize)
    }

    /// Returns the number elements in the sequence.
    ///
    /// This method is equivalent to [`IndexedSeq::len`], but it is provided to
    /// reduce ambiguity in method resolution.
    #[inline]
    pub fn len(&self) -> usize {
        self.n
    }

    /// Returns the upper bound used to build the structure.
    #[inline]
    pub fn upper_bound(&self) -> usize {
        self.u
    }

    /// Replaces the high bits.
    ///
    /// # Safety
    ///
    /// This method is unsafe because it is not possible to guarantee that the
    /// new high bits are identical to the old ones as a bit vector.
    pub unsafe fn map_high_bits<F, H2>(self, func: F) -> EliasFano<H2, L>
    where
        F: FnOnce(H) -> H2,
    {
        EliasFano {
            n: self.n,
            u: self.u,
            l: self.l,
            low_bits: self.low_bits,
            high_bits: func(self.high_bits),
        }
    }

    /// Replaces the low bits.
    ///
    /// # Safety
    ///
    /// This method is unsafe because it is not possible to guarantee that the
    /// new low bits are identical to the old ones as vector.
    pub unsafe fn map_low_bits<F, L2>(self, func: F) -> EliasFano<H, L2>
    where
        F: FnOnce(L) -> L2,
    {
        EliasFano {
            n: self.n,
            u: self.u,
            l: self.l,
            low_bits: func(self.low_bits),
            high_bits: self.high_bits,
        }
    }
}

impl<H: AsRef<[usize]>, L: SliceByValue<Value = usize>> Types for EliasFano<H, L> {
    type Output<'a> = usize;
    type Input = usize;
}

impl<H: AsRef<[usize]> + SelectUnchecked, L: SliceByValue<Value = usize>> IndexedSeq
    for EliasFano<H, L>
{
    #[inline]
    fn len(&self) -> usize {
        self.n
    }

    #[inline(always)]
    unsafe fn get_unchecked(&self, index: usize) -> usize {
        unsafe {
            let high_bits = self.high_bits.select_unchecked(index) - index;
            let low_bits = self.low_bits.get_value_unchecked(index);
            (high_bits << self.l) | low_bits
        }
    }
}

impl<H: AsRef<[usize]> + SelectZeroUnchecked, L: SliceByValue<Value = usize>> IndexedDict
    for EliasFano<H, L>
where
    for<'b> &'b L: IntoUncheckedIterator<Item = usize>,
{
    fn index_of(&self, value: impl Borrow<Self::Input>) -> Option<usize> {
        let value = *value.borrow();
        if value > self.u {
            return None;
        }
        let zeros_to_skip = value >> self.l;
        let bit_pos = if zeros_to_skip == 0 {
            0
        } else {
            unsafe { self.high_bits.select_zero_unchecked(zeros_to_skip - 1) + 1 }
        };

        let mut rank = bit_pos - zeros_to_skip;
        let mut iter = self.low_bits.into_unchecked_iter_from(rank);
        let mut word_idx = bit_pos / (usize::BITS as usize);
        let bits_to_clean = bit_pos % (usize::BITS as usize);

        // SAFETY: we are certainly iterating within the length of the arrays
        // and within the range of the iterator because there is a successor for sure.

        let mut window = unsafe { *self.high_bits.as_ref().get_unchecked(word_idx) }
            & (usize::MAX << bits_to_clean);

        loop {
            while window == 0 {
                word_idx += 1;
                if word_idx >= self.high_bits.as_ref().len() {
                    return None;
                }
                window = unsafe { *self.high_bits.as_ref().get_unchecked(word_idx) };
            }
            // find the lowest bit set index in the word
            let bit_idx = window.trailing_zeros() as usize;
            // compute the global bit index
            let high_bits = (word_idx * usize::BITS as usize) + bit_idx - rank;
            // compose the value
            let res = (high_bits << self.l) | unsafe { iter.next_unchecked() };
            if res == value {
                return Some(rank);
            }
            if res > value {
                return None;
            }

            // clear the lowest bit set
            window &= window - 1;
            rank += 1;
        }
    }
}

#[allow(clippy::collapsible_else_if)]
impl<H: AsRef<[usize]> + SelectZeroUnchecked, L: SliceByValue<Value = usize>> SuccUnchecked
    for EliasFano<H, L>
where
    for<'b> &'b L: IntoUncheckedIterator<Item = usize>,
{
    unsafe fn succ_unchecked<const STRICT: bool>(
        &self,
        value: impl Borrow<Self::Input>,
    ) -> (usize, Self::Output<'_>) {
        let value = *value.borrow();
        let zeros_to_skip = value >> self.l;
        let bit_pos = if zeros_to_skip == 0 {
            0
        } else {
            unsafe { self.high_bits.select_zero_unchecked(zeros_to_skip - 1) + 1 }
        };

        let mut rank = bit_pos - zeros_to_skip;
        let mut iter = self.low_bits.into_unchecked_iter_from(rank);
        let mut word_idx = bit_pos / (usize::BITS as usize);
        let bits_to_clean = bit_pos % (usize::BITS as usize);

        // SAFETY: we are certainly iterating within the length of the arrays
        // and within the range of the iterator because there is a successor for sure.

        let mut window = unsafe { *self.high_bits.as_ref().get_unchecked(word_idx) }
            & (usize::MAX << bits_to_clean);

        loop {
            while window == 0 {
                word_idx += 1;
                debug_assert!(word_idx < self.high_bits.as_ref().len());
                window = unsafe { *self.high_bits.as_ref().get_unchecked(word_idx) };
            }
            // find the lowest bit set index in the word
            let bit_idx = window.trailing_zeros() as usize;
            // compute the global bit index
            let high_bits = (word_idx * usize::BITS as usize) + bit_idx - rank;
            // compose the value
            let res = (high_bits << self.l) | unsafe { iter.next_unchecked() };

            if STRICT {
                if res > value {
                    return (rank, res);
                }
            } else {
                if res >= value {
                    return (rank, res);
                }
            }

            // clear the lowest bit set
            window &= window - 1;
            rank += 1;
        }
    }
}

impl<H: AsRef<[usize]> + SelectUnchecked + SelectZeroUnchecked, L: SliceByValue<Value = usize>> Succ
    for EliasFano<H, L>
where
    for<'b> &'b L: IntoUncheckedIterator<Item = usize>,
{
}

impl<H: AsRef<[usize]> + SelectZeroUnchecked, L: SliceByValue<Value = usize>> EliasFano<H, L>
where
    for<'b> &'b L: IntoUncheckedIterator<Item = usize>,
{
    /// Returns the index of the successor and an iterator starting
    /// at that position.
    ///
    /// This is an efficient fused version that avoids the
    /// `select` operation that [`iter_from`](EliasFano::iter_from)
    /// would perform.
    ///
    /// # Safety
    ///
    /// The successor must exist.
    #[allow(clippy::collapsible_else_if)]
    pub unsafe fn iter_from_succ_unchecked<const STRICT: bool>(
        &self,
        value: usize,
    ) -> (usize, EliasFanoIterator<'_, H, L>) {
        let zeros_to_skip = value >> self.l;
        let bit_pos = if zeros_to_skip == 0 {
            0
        } else {
            unsafe { self.high_bits.select_zero_unchecked(zeros_to_skip - 1) + 1 }
        };

        let mut rank = bit_pos - zeros_to_skip;
        let mut iter = self.low_bits.into_unchecked_iter_from(rank);
        let mut word_idx = bit_pos / (usize::BITS as usize);
        let bits_to_clean = bit_pos % (usize::BITS as usize);

        let mut window = unsafe { *self.high_bits.as_ref().get_unchecked(word_idx) }
            & (usize::MAX << bits_to_clean);

        loop {
            while window == 0 {
                word_idx += 1;
                debug_assert!(word_idx < self.high_bits.as_ref().len());
                window = unsafe { *self.high_bits.as_ref().get_unchecked(word_idx) };
            }
            let bit_idx = window.trailing_zeros() as usize;
            let high_bits = (word_idx * usize::BITS as usize) + bit_idx - rank;
            let res = (high_bits << self.l) | unsafe { iter.next_unchecked() };

            if STRICT {
                if res > value {
                    return (
                        rank,
                        EliasFanoIterator {
                            ef: self,
                            index: rank,
                            word_idx,
                            window,
                            low_bits: self.low_bits.into_unchecked_iter_from(rank),
                        },
                    );
                }
            } else {
                if res >= value {
                    return (
                        rank,
                        EliasFanoIterator {
                            ef: self,
                            index: rank,
                            word_idx,
                            window,
                            low_bits: self.low_bits.into_unchecked_iter_from(rank),
                        },
                    );
                }
            }

            window &= window - 1;
            rank += 1;
        }
    }
}

impl<H: AsRef<[usize]> + SelectUnchecked + SelectZeroUnchecked, L: SliceByValue<Value = usize>>
    EliasFano<H, L>
where
    for<'b> &'b L: IntoUncheckedIterator<Item = usize>,
{
    pub fn iter_from_succ(&self, value: usize) -> Option<(usize, EliasFanoIterator<'_, H, L>)> {
        if self.n == 0 || value > unsafe { IndexedSeq::get_unchecked(self, self.n - 1) } {
            None
        } else {
            Some(unsafe { self.iter_from_succ_unchecked::<false>(value) })
        }
    }

    pub fn iter_from_succ_strict(
        &self,
        value: usize,
    ) -> Option<(usize, EliasFanoIterator<'_, H, L>)> {
        if self.n == 0 || value >= unsafe { IndexedSeq::get_unchecked(self, self.n - 1) } {
            None
        } else {
            Some(unsafe { self.iter_from_succ_unchecked::<true>(value) })
        }
    }
}

impl<H: AsRef<[usize]> + SelectZeroUnchecked, L: SliceByValue<Value = usize>> EliasFano<H, L> {
    /// Returns the index of the successor and a bidirectional iterator
    /// positioned at that index.
    ///
    /// This is an efficient fused version that avoids the `select` operation
    /// that positioning via [`iter_from`](EliasFano::iter_from) would perform.
    ///
    /// # Safety
    ///
    /// The successor must exist.
    #[allow(clippy::collapsible_else_if)]
    pub unsafe fn bidi_iter_from_succ_unchecked<const STRICT: bool>(
        &self,
        value: usize,
    ) -> (usize, EliasFanoBidiIterator<'_, H, L>) {
        let zeros_to_skip = value >> self.l;
        let bit_pos = if zeros_to_skip == 0 {
            0
        } else {
            unsafe { self.high_bits.select_zero_unchecked(zeros_to_skip - 1) + 1 }
        };

        let mut rank = bit_pos - zeros_to_skip;
        let mut word_idx = bit_pos / (usize::BITS as usize);
        let bits_to_clean = bit_pos % (usize::BITS as usize);

        let full_word = unsafe { *self.high_bits.as_ref().get_unchecked(word_idx) };
        let mut window = full_word & (usize::MAX << bits_to_clean);

        loop {
            while window == 0 {
                word_idx += 1;
                debug_assert!(word_idx < self.high_bits.as_ref().len());
                window = unsafe { *self.high_bits.as_ref().get_unchecked(word_idx) };
            }
            let bit_idx = window.trailing_zeros() as usize;
            let high_bits = (word_idx * usize::BITS as usize) + bit_idx - rank;
            let low = unsafe { self.low_bits.get_value_unchecked(rank) };
            let res = (high_bits << self.l) | low;

            if STRICT {
                if res > value {
                    let full_word = unsafe { *self.high_bits.as_ref().get_unchecked(word_idx) };
                    let index_in_word =
                        (full_word & ((1_usize << bit_idx) - 1)).count_ones() as usize;
                    return (
                        rank,
                        EliasFanoBidiIterator {
                            ef: self,
                            index: rank,
                            word_idx,
                            window: full_word,
                            index_in_word,
                        },
                    );
                }
            } else {
                if res >= value {
                    let full_word = unsafe { *self.high_bits.as_ref().get_unchecked(word_idx) };
                    let index_in_word =
                        (full_word & ((1_usize << bit_idx) - 1)).count_ones() as usize;
                    return (
                        rank,
                        EliasFanoBidiIterator {
                            ef: self,
                            index: rank,
                            word_idx,
                            window: full_word,
                            index_in_word,
                        },
                    );
                }
            }

            window &= window - 1;
            rank += 1;
        }
    }
}

impl<H: AsRef<[usize]> + SelectUnchecked + SelectZeroUnchecked, L: SliceByValue<Value = usize>>
    EliasFano<H, L>
{
    /// Returns the index of the successor and a bidirectional iterator
    /// positioned at that index, or `None` if there is no successor.
    pub fn bidi_iter_from_succ(
        &self,
        value: usize,
    ) -> Option<(usize, EliasFanoBidiIterator<'_, H, L>)> {
        if self.n == 0 || value > unsafe { IndexedSeq::get_unchecked(self, self.n - 1) } {
            None
        } else {
            Some(unsafe { self.bidi_iter_from_succ_unchecked::<false>(value) })
        }
    }

    /// Returns the index of the strict successor and a bidirectional iterator
    /// positioned at that index, or `None` if there is no strict successor.
    pub fn bidi_iter_from_succ_strict(
        &self,
        value: usize,
    ) -> Option<(usize, EliasFanoBidiIterator<'_, H, L>)> {
        if self.n == 0 || value >= unsafe { IndexedSeq::get_unchecked(self, self.n - 1) } {
            None
        } else {
            Some(unsafe { self.bidi_iter_from_succ_unchecked::<true>(value) })
        }
    }
}

#[allow(clippy::collapsible_else_if)]
impl<H: AsRef<[usize]> + SelectZeroUnchecked, L: SliceByValue<Value = usize>> PredUnchecked
    for EliasFano<H, L>
where
    for<'b> &'b L: IntoReverseUncheckedIterator<Item = usize>,
{
    unsafe fn pred_unchecked<const STRICT: bool>(
        &self,
        value: impl Borrow<Self::Input>,
    ) -> (usize, Self::Output<'_>) {
        let value = *value.borrow();
        let zeros_to_skip = value >> self.l;
        let mut bit_pos = unsafe { self.high_bits.select_zero_unchecked(zeros_to_skip) } - 1;

        let mut rank = bit_pos - zeros_to_skip;
        let mut iter = self.low_bits.into_rev_unchecked_iter_from(rank + 1);

        // SAFETY: we are certainly iterating within the length of the arrays
        // and within the range of the iterator because there is a predecessor for sure.
        unsafe {
            loop {
                let lower_bits = iter.next_unchecked();
                let mut word_idx = bit_pos / (usize::BITS as usize);
                let bit_idx = bit_pos % (usize::BITS as usize);
                if self.high_bits.as_ref().get_unchecked(word_idx) & (1_usize << bit_idx) == 0 {
                    let mut zeros = bit_idx;
                    let mut window =
                        *self.high_bits.as_ref().get_unchecked(word_idx) & !(usize::MAX << bit_idx);
                    while window == 0 {
                        word_idx -= 1;
                        window = *self.high_bits.as_ref().get_unchecked(word_idx);
                        zeros += usize::BITS as usize;
                    }
                    return (
                        rank,
                        (((usize::BITS as usize) - 1 + bit_pos
                            - zeros
                            - window.leading_zeros() as usize
                            - rank)
                            << self.l)
                            | lower_bits,
                    );
                }

                if STRICT {
                    if lower_bits < value & ((1 << self.l) - 1) {
                        return (rank, ((bit_pos - rank) << self.l) | lower_bits);
                    }
                } else {
                    if lower_bits <= value & ((1 << self.l) - 1) {
                        return (rank, ((bit_pos - rank) << self.l) | lower_bits);
                    }
                }

                bit_pos -= 1;
                rank -= 1;
            }
        }
    }
}

impl<H: AsRef<[usize]> + SelectUnchecked + SelectZeroUnchecked, L: SliceByValue<Value = usize>> Pred
    for EliasFano<H, L>
where
    for<'b> &'b L: IntoReverseUncheckedIterator<Item = usize>,
{
}

impl<H: AsRef<[usize]> + SelectZeroUnchecked, L: SliceByValue<Value = usize>> EliasFano<H, L>
where
    for<'b> &'b L: IntoReverseUncheckedIterator<Item = usize>,
{
    /// Returns the index of the predecessor and a reverse iterator ending
    /// at that position (inclusive).
    ///
    /// The returned reverse iterator will yield the predecessor and all
    /// preceding elements in decreasing order.
    ///
    /// This is an efficient fused version that avoids the `select` operation
    /// that [`rev_iter_from`](EliasFano::rev_iter_from) would perform.
    ///
    /// # Safety
    ///
    /// The predecessor must exist.
    #[allow(clippy::collapsible_else_if)]
    pub unsafe fn rev_iter_from_pred_unchecked<const STRICT: bool>(
        &self,
        value: usize,
    ) -> (usize, EliasFanoRevIterator<'_, H, L>) {
        let zeros_to_skip = value >> self.l;
        let mut bit_pos = unsafe { self.high_bits.select_zero_unchecked(zeros_to_skip) } - 1;

        let mut rank = bit_pos - zeros_to_skip;
        let mut rev_iter = self.low_bits.into_rev_unchecked_iter_from(rank + 1);

        // SAFETY: we are certainly iterating within the length of the arrays
        // and within the range of the iterator because there is a predecessor for sure.
        unsafe {
            loop {
                let _lower_bits = rev_iter.next_unchecked();
                let mut word_idx = bit_pos / (usize::BITS as usize);
                let bit_idx = bit_pos % (usize::BITS as usize);
                if self.high_bits.as_ref().get_unchecked(word_idx) & (1_usize << bit_idx) == 0 {
                    // bit_pos is a zero: the predecessor must be below this
                    // position. Find the highest set bit at or below bit_pos.
                    let mut window =
                        *self.high_bits.as_ref().get_unchecked(word_idx) & !(usize::MAX << bit_idx);
                    while window == 0 {
                        word_idx -= 1;
                        window = *self.high_bits.as_ref().get_unchecked(word_idx);
                    }
                    // The window has all bits in this word up to (but not
                    // including) bit_idx, and includes the predecessor's bit.
                    // This is exactly the right state for the reverse iterator
                    // at position rank + 1.
                    return (
                        rank,
                        EliasFanoRevIterator {
                            ef: self,
                            index: rank + 1,
                            word_idx,
                            window,
                            low_bits: self.low_bits.into_rev_unchecked_iter_from(rank + 1),
                        },
                    );
                }

                if STRICT {
                    if _lower_bits < value & ((1 << self.l) - 1) {
                        // bit_pos is a one and the low bits match: predecessor
                        // is at rank. Build window with bits 0..=bit_idx.
                        let window = *self.high_bits.as_ref().get_unchecked(word_idx)
                            & (usize::MAX >> (usize::BITS as usize - 1 - bit_idx));
                        return (
                            rank,
                            EliasFanoRevIterator {
                                ef: self,
                                index: rank + 1,
                                word_idx,
                                window,
                                low_bits: self.low_bits.into_rev_unchecked_iter_from(rank + 1),
                            },
                        );
                    }
                } else {
                    if _lower_bits <= value & ((1 << self.l) - 1) {
                        let window = *self.high_bits.as_ref().get_unchecked(word_idx)
                            & (usize::MAX >> (usize::BITS as usize - 1 - bit_idx));
                        return (
                            rank,
                            EliasFanoRevIterator {
                                ef: self,
                                index: rank + 1,
                                word_idx,
                                window,
                                low_bits: self.low_bits.into_rev_unchecked_iter_from(rank + 1),
                            },
                        );
                    }
                }

                bit_pos -= 1;
                rank -= 1;
            }
        }
    }
}

impl<H: AsRef<[usize]> + SelectUnchecked + SelectZeroUnchecked, L: SliceByValue<Value = usize>>
    EliasFano<H, L>
where
    for<'b> &'b L: IntoReverseUncheckedIterator<Item = usize>,
{
    /// Returns the index of the predecessor and a reverse iterator ending
    /// at that position (inclusive), or `None` if there is no predecessor.
    ///
    /// The predecessor is the greatest value in the sequence that is less
    /// than or equal to the given value.
    pub fn rev_iter_from_pred(
        &self,
        value: usize,
    ) -> Option<(usize, EliasFanoRevIterator<'_, H, L>)> {
        if self.n == 0 || value < unsafe { IndexedSeq::get_unchecked(self, 0) } {
            None
        } else {
            Some(unsafe { self.rev_iter_from_pred_unchecked::<false>(value) })
        }
    }

    /// Returns the index of the strict predecessor and a reverse iterator
    /// ending at that position (inclusive), or `None` if there is no strict
    /// predecessor.
    ///
    /// The strict predecessor is the greatest value in the sequence that is
    /// strictly less than the given value.
    pub fn rev_iter_from_pred_strict(
        &self,
        value: usize,
    ) -> Option<(usize, EliasFanoRevIterator<'_, H, L>)> {
        if self.n == 0 || value <= unsafe { IndexedSeq::get_unchecked(self, 0) } {
            None
        } else {
            Some(unsafe { self.rev_iter_from_pred_unchecked::<true>(value) })
        }
    }
}

impl<H: AsRef<[usize]> + SelectZeroUnchecked, L: SliceByValue<Value = usize>> EliasFano<H, L> {
    /// Returns the index of the predecessor and a bidirectional iterator
    /// positioned at that index.
    ///
    /// Calling [`next()`](Iterator::next) on the returned iterator will yield
    /// the predecessor; calling [`prev()`](BidiIterator::prev) will yield the
    /// element before the predecessor, if any.
    ///
    /// This is an efficient fused version that avoids the
    /// [`select`](SelectUnchecked::select_unchecked) operation.
    ///
    /// # Safety
    ///
    /// The predecessor must exist.
    #[allow(clippy::collapsible_else_if)]
    pub unsafe fn bidi_iter_from_pred_unchecked<const STRICT: bool>(
        &self,
        value: usize,
    ) -> (usize, EliasFanoBidiIterator<'_, H, L>) {
        let zeros_to_skip = value >> self.l;
        let mut bit_pos = unsafe { self.high_bits.select_zero_unchecked(zeros_to_skip) } - 1;

        let mut rank = bit_pos - zeros_to_skip;

        unsafe {
            loop {
                let mut word_idx = bit_pos / (usize::BITS as usize);
                let bit_idx = bit_pos % (usize::BITS as usize);
                if self.high_bits.as_ref().get_unchecked(word_idx) & (1_usize << bit_idx) == 0 {
                    // bit_pos is a zero: the predecessor must be below this
                    // position. Find the highest set bit at or below bit_pos.
                    let mut masked =
                        *self.high_bits.as_ref().get_unchecked(word_idx) & !(usize::MAX << bit_idx);
                    while masked == 0 {
                        word_idx -= 1;
                        masked = *self.high_bits.as_ref().get_unchecked(word_idx);
                    }
                    // The predecessor's bit is the highest set bit in masked.
                    let pred_bit = usize::BITS as usize - 1 - masked.leading_zeros() as usize;
                    let full_word = *self.high_bits.as_ref().get_unchecked(word_idx);
                    // index_in_word for cursor at rank: ones at positions < pred_bit
                    let index_in_word =
                        (full_word & ((1_usize << pred_bit) - 1)).count_ones() as usize;
                    return (
                        rank,
                        EliasFanoBidiIterator {
                            ef: self,
                            index: rank,
                            word_idx,
                            window: full_word,
                            index_in_word,
                        },
                    );
                }

                let low = self.low_bits.get_value_unchecked(rank);

                if STRICT {
                    if low < value & ((1 << self.l) - 1) {
                        let full_word = *self.high_bits.as_ref().get_unchecked(word_idx);
                        let index_in_word =
                            (full_word & ((1_usize << bit_idx) - 1)).count_ones() as usize;
                        return (
                            rank,
                            EliasFanoBidiIterator {
                                ef: self,
                                index: rank,
                                word_idx,
                                window: full_word,
                                index_in_word,
                            },
                        );
                    }
                } else {
                    if low <= value & ((1 << self.l) - 1) {
                        let full_word = *self.high_bits.as_ref().get_unchecked(word_idx);
                        let index_in_word =
                            (full_word & ((1_usize << bit_idx) - 1)).count_ones() as usize;
                        return (
                            rank,
                            EliasFanoBidiIterator {
                                ef: self,
                                index: rank,
                                word_idx,
                                window: full_word,
                                index_in_word,
                            },
                        );
                    }
                }

                bit_pos -= 1;
                rank -= 1;
            }
        }
    }
}

impl<H: AsRef<[usize]> + SelectUnchecked + SelectZeroUnchecked, L: SliceByValue<Value = usize>>
    EliasFano<H, L>
{
    /// Returns the index of the predecessor and a bidirectional iterator
    /// positioned at that index, or [`None`] if there is no predecessor.
    pub fn bidi_iter_from_pred(
        &self,
        value: usize,
    ) -> Option<(usize, EliasFanoBidiIterator<'_, H, L>)> {
        if self.n == 0 || value < unsafe { IndexedSeq::get_unchecked(self, 0) } {
            None
        } else {
            Some(unsafe { self.bidi_iter_from_pred_unchecked::<false>(value) })
        }
    }

    /// Returns the index of the strict predecessor and a bidirectional
    /// iterator positioned at that index, or [`None`] if there is no strict
    /// predecessor.
    pub fn bidi_iter_from_pred_strict(
        &self,
        value: usize,
    ) -> Option<(usize, EliasFanoBidiIterator<'_, H, L>)> {
        if self.n == 0 || value <= unsafe { IndexedSeq::get_unchecked(self, 0) } {
            None
        } else {
            Some(unsafe { self.bidi_iter_from_pred_unchecked::<true>(value) })
        }
    }
}

impl<H: AsRef<[usize]>, L: SliceByValue<Value = usize>> EliasFano<H, L>
where
    for<'b> &'b L: IntoUncheckedIterator<Item = usize>,
{
    #[inline(always)]
    pub fn iter(&self) -> EliasFanoIterator<'_, H, L> {
        EliasFanoIterator::new(self)
    }
}

impl<H: AsRef<[usize]>, L: SliceByValue<Value = usize>> EliasFano<H, L>
where
    for<'b> &'b L: IntoReverseUncheckedIterator<Item = usize>,
{
    /// Returns a reverse iterator over the values of the sequence, starting
    /// from the last element and going backwards.
    ///
    /// This method does not require [`SelectUnchecked`] on the high bits,
    /// as it finds the last word by scanning from the end of the high-bits
    /// array.
    pub fn rev_iter(&self) -> EliasFanoRevIterator<'_, H, L> {
        let high = self.high_bits.as_ref();
        let (word_idx, window) = if high.is_empty() {
            (0, 0)
        } else {
            let mut word_idx = high.len() - 1;
            let mut window = unsafe { *high.get_unchecked(word_idx) };
            while window == 0 && word_idx > 0 {
                word_idx -= 1;
                window = unsafe { *high.get_unchecked(word_idx) };
            }
            (word_idx, window)
        };
        EliasFanoRevIterator {
            ef: self,
            index: self.n,
            word_idx,
            window,
            low_bits: self.low_bits.into_rev_unchecked_iter_from(self.n),
        }
    }
}

impl<H: AsRef<[usize]> + SelectUnchecked, L: SliceByValue<Value = usize>> EliasFano<H, L>
where
    for<'b> &'b L: IntoUncheckedIterator<Item = usize>,
{
    #[inline(always)]
    pub fn iter_from(&self, from: usize) -> EliasFanoIterator<'_, H, L> {
        EliasFanoIterator::new_from(self, from)
    }
}

impl<H: AsRef<[usize]> + SelectUnchecked, L: SliceByValue<Value = usize>> EliasFano<H, L>
where
    for<'b> &'b L: IntoUncheckedIterator<Item = usize> + IntoReverseUncheckedIterator<Item = usize>,
{
    /// Returns a reverse iterator that yields elements before position `from`
    /// in decreasing order.
    ///
    /// This is equivalent to `self.iter_from(from).rev()`.
    #[inline(always)]
    pub fn rev_iter_from(&self, from: usize) -> EliasFanoRevIterator<'_, H, L> {
        self.iter_from(from).rev()
    }
}

impl<'a, H: AsRef<[usize]> + SelectUnchecked, L: SliceByValue<Value = usize>> IntoIteratorFrom
    for &'a EliasFano<H, L>
where
    for<'b> &'b L: IntoUncheckedIterator<Item = usize>,
{
    type IntoIterFrom = EliasFanoIterator<'a, H, L>;

    #[inline(always)]
    fn into_iter_from(self, from: usize) -> EliasFanoIterator<'a, H, L> {
        EliasFanoIterator::new_from(self, from)
    }
}

impl<'a, H: AsRef<[usize]> + SelectUnchecked, L: SliceByValue<Value = usize>> IntoBidiIterator
    for &'a EliasFano<H, L>
{
    type Item = usize;
    type IntoBidiIter = EliasFanoBidiIterator<'a, H, L>;

    #[inline(always)]
    fn into_bidi_iter(self) -> EliasFanoBidiIterator<'a, H, L> {
        self.into_bidi_iter_from(0)
    }
}

impl<'a, H: AsRef<[usize]> + SelectUnchecked, L: SliceByValue<Value = usize>> IntoBidiIteratorFrom
    for &'a EliasFano<H, L>
{
    type IntoBidiIterFrom = EliasFanoBidiIterator<'a, H, L>;

    #[inline(always)]
    fn into_bidi_iter_from(self, from: usize) -> EliasFanoBidiIterator<'a, H, L> {
        if from > self.n {
            panic!("Index out of bounds: {} > {}", from, self.n);
        }
        if self.n == 0 {
            return EliasFanoBidiIterator {
                ef: self,
                index: 0,
                word_idx: 0,
                window: 0,
                index_in_word: 0,
            };
        }
        // When from == n we use select(n - 1) to find the last element's
        // word, then set index_in_word past all ones in that word.
        let bit_pos = if from == self.n {
            unsafe { self.high_bits.select_unchecked(from - 1) }
        } else {
            unsafe { self.high_bits.select_unchecked(from) }
        };
        let word_idx = bit_pos / (usize::BITS as usize);
        let window = unsafe { *self.high_bits.as_ref().get_unchecked(word_idx) };
        let index_in_word = if from == self.n {
            (window & (usize::MAX >> (usize::BITS as usize - 1 - bit_pos % usize::BITS as usize)))
                .count_ones() as usize
        } else {
            (window & ((1_usize << (bit_pos % usize::BITS as usize)) - 1)).count_ones() as usize
        };
        EliasFanoBidiIterator {
            ef: self,
            index: from,
            word_idx,
            window,
            index_in_word,
        }
    }
}

impl<H: AsRef<[usize]> + SelectUnchecked, L: SliceByValue<Value = usize>> EliasFano<H, L> {
    /// Returns a bidirectional iterator positioned at the first element.
    #[inline(always)]
    pub fn bidi_iter(&self) -> EliasFanoBidiIterator<'_, H, L> {
        self.into_bidi_iter()
    }

    /// Returns a bidirectional iterator positioned at the given index.
    #[inline(always)]
    pub fn bidi_iter_from(&self, from: usize) -> EliasFanoBidiIterator<'_, H, L> {
        self.into_bidi_iter_from(from)
    }
}

// -----------------------------------------------------------------------------
// Value traits

impl<H: AsRef<[usize]> + SelectUnchecked, L: SliceByValue<Value = usize>>
    value_traits::slices::SliceByValue for EliasFano<H, L>
{
    type Value = usize;

    fn len(&self) -> usize {
        self.n
    }
    unsafe fn get_value_unchecked(&self, index: usize) -> Self::Value {
        unsafe { <Self as IndexedSeq>::get_unchecked(self, index) }
    }
}

impl<'a, H: AsRef<[usize]> + SelectUnchecked, L: SliceByValue<Value = usize>>
    value_traits::iter::IterateByValueGat<'a> for EliasFano<H, L>
where
    for<'c> &'c L: IntoUncheckedIterator<Item = usize>,
{
    type Item = usize;
    type Iter = EliasFanoIterator<'a, H, L>;
}

impl<H: AsRef<[usize]> + SelectUnchecked, L: SliceByValue<Value = usize>>
    value_traits::iter::IterateByValue for EliasFano<H, L>
where
    for<'c> &'c L: IntoUncheckedIterator<Item = usize>,
{
    fn iter_value(&self) -> <Self as value_traits::iter::IterateByValueGat<'_>>::Iter {
        self.iter_from(0)
    }
}

impl<'a, H: AsRef<[usize]> + SelectUnchecked, L: SliceByValue<Value = usize>>
    value_traits::iter::IterateByValueFromGat<'a> for EliasFano<H, L>
where
    for<'c> &'c L: IntoUncheckedIterator<Item = usize>,
{
    type Item = usize;
    type IterFrom = EliasFanoIterator<'a, H, L>;
}

impl<H: AsRef<[usize]> + SelectUnchecked, L: SliceByValue<Value = usize>>
    value_traits::iter::IterateByValueFrom for EliasFano<H, L>
where
    for<'b> &'b L: IntoUncheckedIterator<Item = usize>,
{
    fn iter_value_from(
        &self,
        from: usize,
    ) -> <Self as value_traits::iter::IterateByValueGat<'_>>::Iter {
        self.iter_from(from)
    }
}

impl<'a, 'b, H: AsRef<[usize]> + SelectUnchecked, L: SliceByValue<Value = usize>>
    value_traits::iter::IterateByValueGat<'a> for EliasFanoSubsliceImpl<'b, H, L>
where
    for<'c> &'c L: IntoUncheckedIterator<Item = usize>,
{
    type Item = usize;
    type Iter = EliasFanoIterator<'a, H, L>;
}

impl<'a, H: AsRef<[usize]> + SelectUnchecked, L: SliceByValue<Value = usize>>
    value_traits::iter::IterateByValue for EliasFanoSubsliceImpl<'a, H, L>
where
    for<'c> &'c L: IntoUncheckedIterator<Item = usize>,
{
    fn iter_value(&self) -> <Self as value_traits::iter::IterateByValueGat<'_>>::Iter {
        self.slice.iter_from(0)
    }
}

impl<'a, 'b, H: AsRef<[usize]> + SelectUnchecked, L: SliceByValue<Value = usize>>
    value_traits::iter::IterateByValueFromGat<'a> for EliasFanoSubsliceImpl<'b, H, L>
where
    for<'c> &'c L: IntoUncheckedIterator<Item = usize>,
{
    type Item = usize;
    type IterFrom = EliasFanoIterator<'a, H, L>;
}

impl<'a, H: AsRef<[usize]> + SelectUnchecked, L: SliceByValue<Value = usize>>
    value_traits::iter::IterateByValueFrom for EliasFanoSubsliceImpl<'a, H, L>
where
    for<'c> &'c L: IntoUncheckedIterator<Item = usize>,
{
    fn iter_value_from(
        &self,
        from: usize,
    ) -> <Self as value_traits::iter::IterateByValueGat<'_>>::Iter {
        self.slice.iter_from(from)
    }
}

/// An iterator for [`EliasFano`].
#[derive(MemDbg, MemSize)]
pub struct EliasFanoIterator<'a, H: AsRef<[usize]>, L: SliceByValue<Value = usize>>
where
    for<'b> &'b L: IntoUncheckedIterator<Item = usize>,
{
    ef: &'a EliasFano<H, L>,
    /// The index of the next value it will be returned when `next` is called.
    index: usize,
    /// Index of the word loaded in the `word` field.
    word_idx: usize,
    /// Current window on the high bits.
    /// This is an usize because BitVec is implemented only for `Vec<usize>` and `&[usize]`.
    window: usize,
    low_bits: <&'a L as IntoUncheckedIterator>::IntoUncheckedIter,
}

impl<'a, H: AsRef<[usize]>, L: SliceByValue<Value = usize>> EliasFanoIterator<'a, H, L>
where
    for<'b> &'b L: IntoUncheckedIterator<Item = usize>,
{
    pub fn new(ef: &'a EliasFano<H, L>) -> Self {
        let word = if ef.high_bits.as_ref().is_empty() {
            0
        } else {
            unsafe { *ef.high_bits.as_ref().get_unchecked(0) }
        };
        Self {
            ef,
            index: 0,
            word_idx: 0,
            window: word,
            low_bits: ef.low_bits.into_unchecked_iter(),
        }
    }
}

impl<'a, H: AsRef<[usize]> + SelectUnchecked, L: SliceByValue<Value = usize>>
    EliasFanoIterator<'a, H, L>
where
    for<'b> &'b L: IntoUncheckedIterator<Item = usize>,
{
    pub fn new_from(ef: &'a EliasFano<H, L>, start_index: usize) -> Self {
        if start_index > ef.len() {
            panic!("Index out of bounds: {} > {}", start_index, ef.len());
        }
        if start_index == ef.len() {
            return Self {
                ef,
                index: start_index,
                word_idx: 0,
                window: 0,
                low_bits: ef.low_bits.into_unchecked_iter_from(start_index),
            };
        }
        let bit_pos = unsafe { ef.high_bits.select_unchecked(start_index) };
        let word_idx = bit_pos / (usize::BITS as usize);
        let bits_to_clean = bit_pos % (usize::BITS as usize);

        let window = if ef.high_bits.as_ref().is_empty() {
            0
        } else {
            // get the word from the high bits
            let word = unsafe { *ef.high_bits.as_ref().get_unchecked(word_idx) };
            // clean off the bits that we don't care about
            word & (usize::MAX << bits_to_clean)
        };

        Self {
            ef,
            index: start_index,
            word_idx,
            window,
            low_bits: ef.low_bits.into_unchecked_iter_from(start_index),
        }
    }
}

impl<H: AsRef<[usize]>, L: SliceByValue<Value = usize>> Iterator for EliasFanoIterator<'_, H, L>
where
    for<'b> &'b L: IntoUncheckedIterator<Item = usize>,
{
    type Item = usize;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.ef.len() {
            return None;
        }
        // find the next word with zeros
        while self.window == 0 {
            self.word_idx += 1;
            debug_assert!(self.word_idx < self.ef.high_bits.as_ref().len());
            self.window = unsafe { *self.ef.high_bits.as_ref().get_unchecked(self.word_idx) };
        }
        // find the lowest bit set index in the word
        let bit_idx = self.window.trailing_zeros() as usize;
        // compute the global bit index
        let high_bits = (self.word_idx * usize::BITS as usize) + bit_idx - self.index;
        // clear the lowest bit set
        self.window &= self.window - 1;
        // compose the value
        let res = (high_bits << self.ef.l) | unsafe { self.low_bits.next_unchecked() };
        self.index += 1;
        Some(res)
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }

    #[inline(always)]
    fn count(self) -> usize {
        self.ef.len() - self.index
    }

    #[inline(always)]
    fn last(self) -> Option<Self::Item> {
        if self.index >= self.ef.n {
            return None;
        }
        let words = self.ef.high_bits.as_ref();
        let mut word_idx = words.len() - 1;
        // SAFETY: n > 0 implies the high bits contain at least one set bit.
        while unsafe { *words.get_unchecked(word_idx) } == 0 {
            debug_assert!(word_idx > 0);
            word_idx -= 1;
        }
        let word = unsafe { *words.get_unchecked(word_idx) };
        let bit_idx = usize::BITS as usize - 1 - word.leading_zeros() as usize;
        let high_bits = (word_idx * usize::BITS as usize) + bit_idx - (self.ef.n - 1);
        let low = unsafe { self.ef.low_bits.get_value_unchecked(self.ef.n - 1) };
        Some((high_bits << self.ef.l) | low)
    }
}

impl<H: AsRef<[usize]>, L: SliceByValue<Value = usize>> ExactSizeIterator
    for EliasFanoIterator<'_, H, L>
where
    for<'b> &'b L: IntoUncheckedIterator<Item = usize>,
{
    #[inline(always)]
    fn len(&self) -> usize {
        self.ef.len() - self.index
    }
}

impl<H: AsRef<[usize]>, L: SliceByValue<Value = usize>> FusedIterator
    for EliasFanoIterator<'_, H, L>
where
    for<'b> &'b L: IntoUncheckedIterator<Item = usize>,
{
}

impl<'a, H: AsRef<[usize]>, L: SliceByValue<Value = usize>> EliasFanoIterator<'a, H, L>
where
    for<'b> &'b L: IntoUncheckedIterator<Item = usize> + IntoReverseUncheckedIterator<Item = usize>,
{
    /// Converts this forward iterator into a reverse iterator at the current
    /// cursor position.
    ///
    /// The reverse iterator will yield elements before the current position
    /// in decreasing order. The high-bits window is converted using XOR with
    /// the original word, and the low-bits reverse iterator is created from
    /// the current index.
    pub fn rev(self) -> EliasFanoRevIterator<'a, H, L> {
        // When the forward iterator is exhausted (index >= n), the
        // word_idx/window state may not reflect the actual end position
        // (e.g., when created via new_from(ef, n)). We delegate to
        // rev_iter() which correctly scans from the end.
        // This also handles the n == 0 case, since 0 >= 0.
        if self.index >= self.ef.n {
            return self.ef.rev_iter();
        }
        let original = unsafe { *self.ef.high_bits.as_ref().get_unchecked(self.word_idx) };
        EliasFanoRevIterator {
            ef: self.ef,
            index: self.index,
            word_idx: self.word_idx,
            window: self.window ^ original,
            low_bits: self.ef.low_bits.into_rev_unchecked_iter_from(self.index),
        }
    }
}

/// A reverse iterator for [`EliasFano`].
///
/// Instead of scanning bits from right to left (using [`trailing_zeros`](usize::trailing_zeros)),
/// it scans from left to right (using [`leading_zeros`](usize::leading_zeros)),
/// and accesses low bits through a reverse unchecked iterator.
#[derive(MemDbg, MemSize)]
pub struct EliasFanoRevIterator<'a, H: AsRef<[usize]>, L: SliceByValue<Value = usize>>
where
    for<'b> &'b L: IntoReverseUncheckedIterator<Item = usize>,
{
    ef: &'a EliasFano<H, L>,
    /// The index of the next value that will be returned when `next` is
    /// called, plus one; that is, `next` will return the value at position
    /// `index - 1` and then decrement `index`.
    index: usize,
    /// Index of the word loaded in the `window` field.
    word_idx: usize,
    /// Current window on the high bits.
    window: usize,
    low_bits: <&'a L as IntoReverseUncheckedIterator>::IntoRevUncheckedIter,
}

impl<'a, H: AsRef<[usize]>, L: SliceByValue<Value = usize>> EliasFanoRevIterator<'a, H, L>
where
    for<'b> &'b L: IntoUncheckedIterator<Item = usize> + IntoReverseUncheckedIterator<Item = usize>,
{
    /// Converts this reverse iterator back into a forward iterator at the
    /// current cursor position.
    ///
    /// The forward iterator will yield elements from the current position
    /// onward in increasing order. The high-bits window is converted using
    /// XOR with the original word, and the low-bits forward iterator is
    /// created from the current index.
    pub fn rev(self) -> EliasFanoIterator<'a, H, L> {
        let window = if self.ef.high_bits.as_ref().is_empty() {
            self.window
        } else {
            self.window ^ unsafe { *self.ef.high_bits.as_ref().get_unchecked(self.word_idx) }
        };
        EliasFanoIterator {
            ef: self.ef,
            index: self.index,
            word_idx: self.word_idx,
            window,
            low_bits: self.ef.low_bits.into_unchecked_iter_from(self.index),
        }
    }
}

impl<H: AsRef<[usize]>, L: SliceByValue<Value = usize>> Iterator for EliasFanoRevIterator<'_, H, L>
where
    for<'b> &'b L: IntoReverseUncheckedIterator<Item = usize>,
{
    type Item = usize;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.index == 0 {
            return None;
        }
        while self.window == 0 {
            self.word_idx -= 1;
            self.window = unsafe { *self.ef.high_bits.as_ref().get_unchecked(self.word_idx) };
        }
        let bit_idx = usize::BITS as usize - 1 - self.window.leading_zeros() as usize;
        self.window ^= 1 << bit_idx;
        self.index -= 1;
        let high_bits = (self.word_idx * usize::BITS as usize) + bit_idx - self.index;
        let low = unsafe { self.low_bits.next_unchecked() };
        Some((high_bits << self.ef.l) | low)
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }

    #[inline(always)]
    fn count(self) -> usize {
        self.index
    }

    #[inline(always)]
    fn last(self) -> Option<Self::Item> {
        if self.index == 0 {
            return None;
        }
        let words = self.ef.high_bits.as_ref();
        let mut word_idx = 0;
        // SAFETY: index > 0 implies the high bits contain at least one set bit.
        while unsafe { *words.get_unchecked(word_idx) } == 0 {
            debug_assert!(word_idx + 1 < words.len());
            word_idx += 1;
        }
        let bit_idx = unsafe { *words.get_unchecked(word_idx) }.trailing_zeros() as usize;
        let high_bits = (word_idx * usize::BITS as usize) + bit_idx;
        let low = unsafe { self.ef.low_bits.get_value_unchecked(0) };
        Some((high_bits << self.ef.l) | low)
    }
}

impl<H: AsRef<[usize]>, L: SliceByValue<Value = usize>> ExactSizeIterator
    for EliasFanoRevIterator<'_, H, L>
where
    for<'b> &'b L: IntoReverseUncheckedIterator<Item = usize>,
{
    #[inline(always)]
    fn len(&self) -> usize {
        self.index
    }
}

impl<H: AsRef<[usize]>, L: SliceByValue<Value = usize>> FusedIterator
    for EliasFanoRevIterator<'_, H, L>
where
    for<'b> &'b L: IntoReverseUncheckedIterator<Item = usize>,
{
}

impl<'a, H: AsRef<[usize]>, L: SliceByValue<Value = usize>> IntoIterator for &'a EliasFano<H, L>
where
    for<'b> &'b L: IntoUncheckedIterator<Item = usize>,
{
    type Item = usize;
    type IntoIter = EliasFanoIterator<'a, H, L>;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        EliasFanoIterator::new(self)
    }
}

/// A bidirectional iterator (cursor) for [`EliasFano`].
///
/// Unlike [`EliasFanoIterator`] and [`EliasFanoRevIterator`], this cursor
/// does not clear bits from the current word. Instead, it uses
/// [`select_in_word`](SelectInWord::select_in_word) to find the relevant bit
/// on each call to [`next`](EliasFanoBidiIterator::next) or
/// [`prev`](EliasFanoBidiIterator::prev). Low bits are accessed via random
/// access ([`get_value_unchecked`](SliceByValue::get_value_unchecked)).
///
/// The cursor position `index` ranges from 0 to *n*. Calling `next()` yields
/// element `index` and increments the cursor; calling `prev()` yields element
/// `index - 1` and decrements it.
#[derive(MemDbg, MemSize)]
pub struct EliasFanoBidiIterator<'a, H: AsRef<[usize]>, L: SliceByValue<Value = usize>> {
    ef: &'a EliasFano<H, L>,
    /// Cursor position: `next()` yields element `index`, `prev()` yields
    /// element `index - 1`.
    index: usize,
    /// Index of the word loaded in `window`.
    word_idx: usize,
    /// The full, unmodified word from `high_bits[word_idx]`.
    window: usize,
    /// Rank of the cursor within the current word: the number of ones in
    /// `window` that correspond to elements at positions < `index`.
    index_in_word: usize,
}

impl<H: AsRef<[usize]>, L: SliceByValue<Value = usize>> Iterator
    for EliasFanoBidiIterator<'_, H, L>
{
    type Item = usize;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.ef.n {
            return None;
        }
        // Advance to the next word if we've exhausted the ones in this word.
        while self.index_in_word >= self.window.count_ones() as usize {
            self.index_in_word -= self.window.count_ones() as usize;
            self.word_idx += 1;
            self.window = unsafe { *self.ef.high_bits.as_ref().get_unchecked(self.word_idx) };
        }
        let bit_idx = self.window.select_in_word(self.index_in_word);
        let high_bits = (self.word_idx * usize::BITS as usize) + bit_idx - self.index;
        let low = unsafe { self.ef.low_bits.get_value_unchecked(self.index) };
        self.index += 1;
        self.index_in_word += 1;
        Some((high_bits << self.ef.l) | low)
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.ef.n - self.index;
        (remaining, Some(remaining))
    }

    #[inline(always)]
    fn count(self) -> usize {
        self.ef.n - self.index
    }

    #[inline(always)]
    fn last(self) -> Option<Self::Item> {
        if self.index >= self.ef.n {
            return None;
        }
        let words = self.ef.high_bits.as_ref();
        let mut word_idx = words.len() - 1;
        // SAFETY: n > 0 implies the high bits contain at least one set bit.
        while unsafe { *words.get_unchecked(word_idx) } == 0 {
            debug_assert!(word_idx > 0);
            word_idx -= 1;
        }
        let word = unsafe { *words.get_unchecked(word_idx) };
        let bit_idx = usize::BITS as usize - 1 - word.leading_zeros() as usize;
        let high_bits = (word_idx * usize::BITS as usize) + bit_idx - (self.ef.n - 1);
        let low = unsafe { self.ef.low_bits.get_value_unchecked(self.ef.n - 1) };
        Some((high_bits << self.ef.l) | low)
    }
}

impl<H: AsRef<[usize]>, L: SliceByValue<Value = usize>> ExactSizeIterator
    for EliasFanoBidiIterator<'_, H, L>
{
    #[inline(always)]
    fn len(&self) -> usize {
        self.ef.n - self.index
    }
}

impl<H: AsRef<[usize]>, L: SliceByValue<Value = usize>> FusedIterator
    for EliasFanoBidiIterator<'_, H, L>
{
}

impl<H: AsRef<[usize]>, L: SliceByValue<Value = usize>> BidiIterator
    for EliasFanoBidiIterator<'_, H, L>
{
    type PrevIter = PrevIter<Self>;

    #[inline(always)]
    fn prev_iter(self) -> PrevIter<Self> {
        PrevIter(self)
    }

    #[inline(always)]
    fn prev(&mut self) -> Option<usize> {
        if self.index == 0 {
            return None;
        }
        // Move to the previous word if we're at the start of this word.
        while self.index_in_word == 0 {
            self.word_idx -= 1;
            self.window = unsafe { *self.ef.high_bits.as_ref().get_unchecked(self.word_idx) };
            self.index_in_word = self.window.count_ones() as usize;
        }
        self.index -= 1;
        self.index_in_word -= 1;
        let bit_idx = self.window.select_in_word(self.index_in_word);
        let high_bits = (self.word_idx * usize::BITS as usize) + bit_idx - self.index;
        let low = unsafe { self.ef.low_bits.get_value_unchecked(self.index) };
        Some((high_bits << self.ef.l) | low)
    }

    #[inline(always)]
    fn prev_size_hint(&self) -> (usize, Option<usize>) {
        (self.index, Some(self.index))
    }

    #[inline(always)]
    fn prev_count(self) -> usize {
        self.index
    }

    #[inline(always)]
    fn prev_last(self) -> Option<usize> {
        if self.index == 0 {
            return None;
        }
        let words = self.ef.high_bits.as_ref();
        let mut word_idx = 0;
        // SAFETY: index > 0 implies the high bits contain at least one set bit.
        while unsafe { *words.get_unchecked(word_idx) } == 0 {
            debug_assert!(word_idx + 1 < words.len());
            word_idx += 1;
        }
        let bit_idx = unsafe { *words.get_unchecked(word_idx) }.trailing_zeros() as usize;
        let high_bits = (word_idx * usize::BITS as usize) + bit_idx;
        let low = unsafe { self.ef.low_bits.get_value_unchecked(0) };
        Some((high_bits << self.ef.l) | low)
    }
}

impl<H: AsRef<[usize]>, L: SliceByValue<Value = usize>> ExactSizeBidiIterator
    for EliasFanoBidiIterator<'_, H, L>
{
    #[inline(always)]
    fn prev_len(&self) -> usize {
        self.index
    }
}

impl<H: AsRef<[usize]>, L: SliceByValue<Value = usize>> FusedBidiIterator
    for EliasFanoBidiIterator<'_, H, L>
{
}

/// Convenience constructor that iterates over a slice.
///
/// Note that this implementation requires a first scan to check monotonicity
/// and find the maximum value, but then it uses
/// [`EliasFanoBuilder::push_unchecked`], thus partially compensating for the
/// cost of the first scan.
impl<A: AsRef<[usize]>> From<A> for EliasFano {
    fn from(values: A) -> Self {
        let values = values.as_ref();
        let mut max = 0;
        let mut prev = 0;
        for &value in values {
            if value < prev {
                panic!("The values provided are not monotone: {} < {}", value, prev);
            }
            max = max.max(value);
            prev = value;
        }
        let mut builder = EliasFanoBuilder::new(values.len(), max);
        for &value in values {
            // SAFETY: pre-scan checked monotonicity and max.
            unsafe {
                builder.push_unchecked(value);
            }
        }
        builder.build()
    }
}

/// A sequential builder for [`EliasFano`].
///
/// After creating an instance, you can use [`EliasFanoBuilder::push`] to add
/// new values, and then call [`EliasFanoBuilder::build`] to create the
/// [`EliasFano`] instance.
///
/// # Examples
///
/// ```rust
/// # use sux::dict::EliasFanoBuilder;
/// let mut efb = EliasFanoBuilder::new(4, 10);
///
/// efb.push(0);
/// efb.push(2);
/// efb.push(8);
/// efb.push(10);
///
/// let ef = efb.build();
/// let mut iter = ef.iter();
/// assert_eq!(iter.next(), Some(0));
/// assert_eq!(iter.next(), Some(2));
/// assert_eq!(iter.next(), Some(8));
/// assert_eq!(iter.next(), Some(10));
/// assert_eq!(iter.next(), None);
/// ```
#[derive(Debug, Clone, MemDbg, MemSize)]
pub struct EliasFanoBuilder {
    n: usize,
    u: usize,
    l: usize,
    low_bits: BitFieldVec,
    high_bits: BitVec,
    last_value: usize,
    count: usize,
}

impl EliasFanoBuilder {
    /// Creates a builder for an [`EliasFano`] containing
    /// `n` numbers smaller than or equal to `u`.
    ///
    /// # Panic
    ///
    /// When any of the underlying structures would exceed `usize` in length.
    pub fn new(n: usize, u: usize) -> Self {
        let l = if n > 0 && u >= n {
            (u as f64 / n as f64).log2().floor() as usize
        } else {
            0
        };

        let num_high_bits = n
            .checked_add(1)
            .unwrap_or_else(|| panic!("n ({n}) is too large"))
            .checked_add(u >> l)
            .unwrap_or_else(|| panic!("n ({n}) and/or u ({u}) is too large"));
        Self {
            n,
            u,
            l,
            low_bits: BitFieldVec::new(l, n),
            high_bits: BitVec::new(num_high_bits),
            last_value: 0,
            count: 0,
        }
    }
    /// Adds a new value to the builder.
    ///
    /// # Panic
    /// May panic if the value is smaller than the last provided
    /// value, or if too many values are provided.
    pub fn push(&mut self, value: usize) {
        if self.count == self.n {
            panic!("Too many values");
        }
        if value > self.u {
            panic!("Value too large: {} > {}", value, self.u);
        }
        if value < self.last_value {
            panic!(
                "The values provided are not monotone: {} < {}",
                value, self.last_value
            );
        }
        unsafe {
            self.push_unchecked(value);
        }
    }

    /// # Safety
    ///
    /// Values passed to this function must be smaller than or equal `u` and must be monotone.
    /// Moreover, the function should not be called more than `n` times.
    pub unsafe fn push_unchecked(&mut self, value: usize) {
        let low = value & ((1 << self.l) - 1);
        self.low_bits.set_value(self.count, low);

        let high = (value >> self.l) + self.count;
        self.high_bits.set(high, true);

        self.count += 1;
        self.last_value = value;
    }

    /// Returns the numbers of values added so far.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Builds an Elias-Fano structure.
    ///
    /// The resulting structure has no selection structure attached. To use it
    /// properly, you need to call [`EliasFano::map_high_bits`] to add to the
    /// high bits a selection structure.
    ///
    /// Usually, however, the default implementations returned by the
    /// [`build_with_seq`](EliasFanoBuilder::build_with_seq),
    /// [`build_with_dict`](EliasFanoBuilder::build_with_dict), and
    /// [`build_with_seq_and_dict`](EliasFanoBuilder::build_with_seq_and_dict)
    /// methods are more convenient.
    pub fn build(self) -> EliasFano {
        assert!(
            self.count == self.n,
            "The declared size ({}) is not equal to the number of values ({})",
            self.n,
            self.count
        );
        let high_bits: BitVec<Box<[usize]>> = self.high_bits.into();
        EliasFano {
            n: self.n,
            u: self.u,
            l: self.l,
            low_bits: self.low_bits.into(),
            // SAFETY: n is the number of ones in the high_bits.
            high_bits,
        }
    }

    /// Builds an Elias-Fano structure with constant-time access, using
    /// default values.
    ///
    /// The resulting structure implements [`IndexedSeq`], but not [`IndexedDict`],
    /// [`Succ`], or [`Pred`].
    pub fn build_with_seq(self) -> EfSeq {
        let ef = self.build();
        unsafe { ef.map_high_bits(SelectAdaptConst::<_, _, 12, 3>::new) }
    }

    /// Builds an Elias-Fano structure with constant-time indexing, using
    /// default values.
    ///
    /// The resulting structure implements [`SuccUnchecked`], and [`PredUnchecked`],
    /// but not [`IndexedSeq`].
    pub fn build_with_dict(self) -> EfDict {
        let ef = self.build();
        unsafe { ef.map_high_bits(SelectZeroAdaptConst::<_, _, 12, 3>::new) }
    }

    /// Builds an Elias-Fano structure with constant-time access and indexing,
    /// using default values.
    ///
    /// The resulting structure implements [`IndexedDict`], [`Succ`],
    /// [`Pred`], and [`IndexedSeq`].
    pub fn build_with_seq_and_dict(self) -> EfSeqDict {
        let ef = self.build();
        unsafe {
            ef.map_high_bits(SelectAdaptConst::<_, _, 12, 3>::new)
                .map_high_bits(SelectZeroAdaptConst::<_, _, 12, 3>::new)
        }
    }
}

impl Extend<usize> for EliasFanoBuilder {
    fn extend<T: IntoIterator<Item = usize>>(&mut self, iter: T) {
        for value in iter {
            self.push(value);
        }
    }
}

/// A concurrent builder for [`EliasFano`].
///
/// After creating an instance, you can use [`EliasFanoConcurrentBuilder::set`]
/// to set the values concurrently. However, this operation is inherently
/// unsafe as no check is performed on the provided data (e.g., duplicate
/// indices and lack of monotonicity are not detected).
///
/// # Examples
///
/// ```rust
/// # use sux::dict::EliasFanoConcurrentBuilder;
/// let mut efcb = EliasFanoConcurrentBuilder::new(4, 10);
/// std::thread::scope(|s| {
///     s.spawn(|| { unsafe { efcb.set(0, 0); } });
///     s.spawn(|| { unsafe { efcb.set(1, 2); } });
///     s.spawn(|| { unsafe { efcb.set(2, 8); } });
///     s.spawn(|| { unsafe { efcb.set(3, 10); } });
/// });
///
/// let ef = efcb.build();
/// let mut iter = ef.iter();
/// assert_eq!(iter.next(), Some(0));
/// assert_eq!(iter.next(), Some(2));
/// assert_eq!(iter.next(), Some(8));
/// assert_eq!(iter.next(), Some(10));
/// assert_eq!(iter.next(), None);
/// ```

#[derive(MemDbg, MemSize)]
pub struct EliasFanoConcurrentBuilder {
    n: usize,
    u: usize,
    l: usize,
    low_bits: AtomicBitFieldVec,
    high_bits: AtomicBitVec,
}

impl EliasFanoConcurrentBuilder {
    /// Creates a concurrent builder for a sequence containing `n` nonnegative
    /// numbers smaller than or equal to `u`.
    pub fn new(n: usize, u: usize) -> Self {
        let l = if n > 0 && u >= n {
            (u as f64 / n as f64).log2().floor() as usize
        } else {
            0
        };

        Self {
            u,
            n,
            l,
            low_bits: AtomicBitFieldVec::new(l, n),
            high_bits: AtomicBitVec::new(n + (u >> l) + 1),
        }
    }

    /// Sets a value concurrently.
    ///
    /// # Safety
    /// - All indices must be distinct.
    /// - All values must be smaller than or equal to `u`.
    /// - All indices must be smaller than `n`.
    /// - You must call this function exactly `n` times.
    pub unsafe fn set(&self, index: usize, value: usize) {
        let low = value & ((1 << self.l) - 1);
        // Note that the concurrency guarantees of BitFieldVec
        // are sufficient for us.
        unsafe {
            self.low_bits
                .set_atomic_unchecked(index, low, Ordering::Relaxed)
        };

        let high = (value >> self.l) + index;
        self.high_bits.set(high, true, Ordering::Relaxed);
    }

    /// Builds an Elias-Fano structure.
    ///
    /// The resulting structure has no selection structure attached. To use it
    /// properly, you need to call [`EliasFano::map_high_bits`] to add to the
    /// high bits a selection structure.
    ///
    /// Usually, however, the default implementations returned by the
    /// [`build_with_seq`](EliasFanoConcurrentBuilder::build_with_seq),
    /// [`build_with_dict`](EliasFanoConcurrentBuilder::build_with_dict), and
    /// [`build_with_seq_and_dict`](EliasFanoConcurrentBuilder::build_with_seq_and_dict)
    /// methods are more convenient.
    pub fn build(self) -> EliasFano {
        let high_bits: BitVec<Box<[usize]>> = self.high_bits.into();
        let low_bits: BitFieldVec<usize, Vec<usize>> = self.low_bits.into();
        let low_bits: BitFieldVec<usize, Box<[usize]>> = low_bits.into();
        EliasFano {
            n: self.n,
            u: self.u,
            l: self.l,
            low_bits,
            high_bits,
        }
    }

    /// Builds an Elias-Fano structure with constant-time access, using
    /// default values.
    ///
    /// The resulting structure implements [`IndexedSeq`], but not [`IndexedDict`],
    /// [`Succ`], or [`Pred`].
    pub fn build_with_seq(self) -> EfSeq {
        let ef = self.build();
        unsafe { ef.map_high_bits(SelectAdaptConst::<_, _, 12, 3>::new) }
    }

    /// Builds an Elias-Fano structure with constant-time indexing, using
    /// default values.
    ///
    /// The resulting structure implements [`IndexedDict`], [`SuccUnchecked`], and [`PredUnchecked`],
    /// but not [`IndexedSeq`].
    pub fn build_with_dict(self) -> EfDict {
        let ef = self.build();
        unsafe { ef.map_high_bits(SelectZeroAdaptConst::<_, _, 12, 3>::new) }
    }

    /// Builds an Elias-Fano structure with constant-time access and indexing,
    /// using default values.
    ///
    /// The resulting structure implements [`IndexedDict`], [`Succ`],
    /// [`Pred`], and [`IndexedSeq`].
    pub fn build_with_seq_and_dict(self) -> EfSeqDict {
        let ef = self.build();
        unsafe {
            ef.map_high_bits(SelectAdaptConst::<_, _, 12, 3>::new)
                .map_high_bits(SelectZeroAdaptConst::<_, _, 12, 3>::new)
        }
    }
}
