/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Bit vector implementations.
//!
//! There are two flavors: [`BitVec`], a mutable bit vector, and
//! [`AtomicBitVec`], a mutable, thread-safe bit vector.
//!
//! Operations on these structures are provided by the extension traits
//! [`BitVecOps`], [`BitVecOpsMut`], and [`AtomicBitVecOps`],
//! which must be pulled in scope as needed. There are also
//! operations that are specific to certain implementations, such as
//! [`push`](BitVec::push).
//!
//! These flavors depend on a backend, and presently we provide:
//!
//! - `BitVec<Vec<usize>>`: a mutable, growable and resizable bit vector;
//! - `BitVec<AsRef<[usize]>>`: an immutable bit vector, useful for
//!   [ε-serde](https://crates.io/crates/epserde) support;
//! - `BitVec<AsRef<[usize]> + AsMut<[usize]>>`: a mutable (but not resizable)
//!   bit vector;
//! - `AtomicBitVec<AsRef<[AtomicUsize]>>`: a thread-safe, mutable (but not
//!   resizable) bit vector.
//!
//! Note that nothing is assumed about the content of the backend outside the
//! bits of the bit vector. Moreover, the content of the backend outside of
//! the bit vector is never modified by the methods of this structure.
//!
//! It is possible to juggle between all flavors using [`From`]/[`Into`], and
//! with [`TryFrom`]/[`TryInto`] when going [from a non-atomic to an atomic bit
//! vector](BitVec#impl-TryFrom%3CBitVec%3C%26%5BW%5D%3E%3E-for-AtomicBitVec%3C%26%5B%3CW+as+IntoAtomic%3E::AtomicType%5D%3E).
//!
//! # Examples
//!
//! ```rust
//! use sux::prelude::*;
//! use sux::traits::bit_vec_ops::*;
//! use std::sync::atomic::Ordering;
//!
//! // Convenience macro
//! let b = bit_vec![0, 1, 0, 1, 1, 0, 1, 0];
//! assert_eq!(b.len(), 8);
//! // Not constant time
//! assert_eq!(b.count_ones(), 4);
//! assert_eq!(b[0], false);
//! assert_eq!(b[1], true);
//! assert_eq!(b[2], false);
//!
//! let b: AddNumBits<_> = b.into();
//! // Constant time, but now b is immutable
//! assert_eq!(b.num_ones(), 4);
//!
//! let mut b = BitVec::new(0);
//! b.push(true);
//! b.push(false);
//! b.push(true);
//! assert_eq!(b.len(), 3);
//!
//! // Let's make it atomic
//! let mut a: AtomicBitVec = b.into();
//! a.set(1, true, Ordering::Relaxed);
//! assert!(a.get(0, Ordering::Relaxed));
//!
//! // Back to normal, but immutable size
//! let b: BitVec<Vec<usize>> = a.into();
//! let mut b: BitVec<Box<[usize]>> = b.into();
//! b.set(2, false);
//!
//! // If we create an artificially dirty bit vector, everything still works.
//! let ones = [usize::MAX; 2];
//! assert_eq!(unsafe { BitVec::from_raw_parts(ones, 1) }.count_ones(), 1);
//! ```

use crate::traits::{AtomicBitIter, AtomicBitVecOps, BitIter, BitVecOps, BitVecOpsMut};
use common_traits::{IntoAtomic, SelectInWord};
#[allow(unused_imports)] // this is in the std prelude but not in no_std!
use core::borrow::BorrowMut;
use core::fmt;
use mem_dbg::*;
use std::{
    ops::Index,
    sync::atomic::{AtomicUsize, Ordering},
};

use crate::{
    traits::{BITS, rank_sel::*},
    utils::{
        CannotCastToAtomicError, transmute_boxed_slice_from_atomic,
        transmute_boxed_slice_into_atomic, transmute_vec_from_atomic, transmute_vec_into_atomic,
    },
};

/// A bit vector.
///
/// Instances can be created using [`new`](BitVec::new),
/// [`with_value`](BitVec::with_value), with the convenience macro
/// [`bit_vec!`](macro@crate::bits::bit_vec), or with a [`FromIterator`
/// implementation](#impl-FromIterator<bool>-for-BitVec).
///
/// See the [module documentation](mod@crate::bits::bit_vec) for more details.
#[derive(Debug, Clone, Copy, MemDbg, MemSize)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BitVec<B = Vec<usize>> {
    bits: B,
    len: usize,
}

/// Convenient, [`vec!`](vec!)-like macro to initialize bit vectors.
///
/// - `bit_vec![]` creates an empty bit vector.
///
/// - `bit_vec![false; n]` or `bit_vec![0; n]` creates a bit vector of length
///   `n` with all bits set to `false`.
///
/// - `bit_vec![true; n]` or `bit_vec![1; n]` creates a bit vector of length `n`
///   with all bits set to `true`.
///
/// - `bit_vec![b₀, b₁, b₂, …]` creates a bit vector with the specified bits,
///   where each `bᵢ` can be any expression that evaluates to a boolean or integer
///   (0 for `false`, non-zero for `true`).
///
/// # Examples
///
/// ```rust
/// use sux::prelude::*;
/// use sux::traits::BitVecOps;
///
/// // Empty bit vector
/// let b = bit_vec![];
/// assert_eq!(b.len(), 0);
///
/// // 10 bits set to true
/// let b = bit_vec![true; 10];
/// assert_eq!(b.len(), 10);
/// assert_eq!(b.iter().all(|x| x), true);
/// let b = bit_vec![1; 10];
/// assert_eq!(b.len(), 10);
/// assert_eq!(b.iter().all(|x| x), true);
///
/// // 10 bits set to false
/// let b = bit_vec![false; 10];
/// assert_eq!(b.len(), 10);
/// assert_eq!(b.iter().any(|x| x), false);
/// let b = bit_vec![0; 10];
/// assert_eq!(b.len(), 10);
/// assert_eq!(b.iter().any(|x| x), false);
///
/// // Bit list
/// let b = bit_vec![0, 1, 0, 1, 0, 0];
/// assert_eq!(b.len(), 6);
/// assert_eq!(b[0], false);
/// assert_eq!(b[1], true);
/// assert_eq!(b[2], false);
/// assert_eq!(b[3], true);
/// assert_eq!(b[4], false);
/// assert_eq!(b[5], false);
/// ```
#[macro_export]
macro_rules! bit_vec {
    () => {
        $crate::bits::BitVec::new(0)
    };
    (false; $n:expr) => {
        $crate::bits::BitVec::new($n)
    };
    (0; $n:expr) => {
        $crate::bits::BitVec::new($n)
    };
    (true; $n:expr) => {
        {
            $crate::bits::BitVec::with_value($n, true)
        }
    };
    (1; $n:expr) => {
        {
            $crate::bits::BitVec::with_value($n, true)
        }
    };
    ($($x:expr),+ $(,)?) => {
        {
            let mut b = $crate::bits::BitVec::with_capacity([$($x),+].len());
            $( b.push($x != 0); )*
            b
        }
    };
}

impl<B> BitVec<B> {
    /// Returns the number of bits in the bit vector.
    ///
    /// This method is equivalent to [`BitLength::len`], but it is provided to
    /// reduce ambiguity in method resolution.
    #[inline(always)]
    pub fn len(&self) -> usize {
        BitLength::len(self)
    }

    /// # Safety
    /// `len` must be between 0 (included) and the number of
    /// bits in `bits` (included).
    #[inline(always)]
    pub unsafe fn from_raw_parts(bits: B, len: usize) -> Self {
        Self { bits, len }
    }

    #[inline(always)]
    pub fn into_raw_parts(self) -> (B, usize) {
        (self.bits, self.len)
    }

    #[inline(always)]
    /// Modify the bit vector in place.
    ///
    ///
    /// # Safety
    /// This is unsafe because it's the caller's responsibility to ensure that
    /// that the length is compatible with the modified bits.
    pub unsafe fn map<B2>(self, f: impl FnOnce(B) -> B2) -> BitVec<B2> {
        BitVec {
            bits: f(self.bits),
            len: self.len,
        }
    }
}

impl BitVec<Vec<usize>> {
    /// Creates a new bit vector of length `len` initialized to `false`.
    pub fn new(len: usize) -> Self {
        Self::with_value(len, false)
    }

    /// Creates a new bit vector of length `len` initialized to `value`.
    pub fn with_value(len: usize, value: bool) -> Self {
        let n_of_words = len.div_ceil(BITS);
        let extra_bits = (n_of_words * BITS) - len;
        let word_value = if value { !0 } else { 0 };
        let mut bits = vec![word_value; n_of_words];
        if extra_bits > 0 {
            let last_word_value = word_value >> extra_bits;
            bits[n_of_words - 1] = last_word_value;
        }
        Self { bits, len }
    }

    /// Creates a new zero-length bit vector of given capacity.
    ///
    /// Note that the capacity will be rounded up to a multiple of the word
    /// size.
    pub fn with_capacity(capacity: usize) -> Self {
        let n_of_words = capacity.div_ceil(BITS);
        Self {
            bits: Vec::with_capacity(n_of_words),
            len: 0,
        }
    }

    /// Returns the current capacity of this bit vector.
    pub fn capacity(&self) -> usize {
        self.bits.capacity() * BITS
    }

    /// Appends a bit to the end of this bit vector.
    pub fn push(&mut self, b: bool) {
        if self.bits.len() * BITS == self.len {
            self.bits.push(0);
        }
        let word_index = self.len / BITS;
        let bit_index = self.len % BITS;
        // Clear bit
        self.bits[word_index] &= !(1 << bit_index);
        // Set bit
        self.bits[word_index] |= (b as usize) << bit_index;
        self.len += 1;
    }

    /// Removes the last bit from the bit vector and returns it, or `None` if it
    /// is empty.
    pub fn pop(&mut self) -> Option<bool> {
        if self.len == 0 {
            return None;
        }
        let last_pos = self.len - 1;
        let result = unsafe { self.get_unchecked(last_pos) };
        self.len = last_pos;
        Some(result)
    }

    /// Resizes the bit vector in place, extending it with `value` if it is
    /// necessary.
    pub fn resize(&mut self, new_len: usize, value: bool) {
        if new_len > self.len {
            let old_len = self.len;
            let old_word = old_len / BITS;
            let old_bit = old_len % BITS;
            let word_value = if value { !0usize } else { 0 };

            self.bits.resize(new_len.div_ceil(BITS), word_value);

            // Handle the partial word at old_len, then fill all
            // remaining words (which may contain stale data from
            // previous truncations).
            if old_bit != 0 {
                let mask = !0usize << old_bit;
                self.bits[old_word] = (self.bits[old_word] & !mask) | (word_value & mask);
                self.bits[old_word + 1..].fill(word_value);
            } else {
                self.bits[old_word..].fill(word_value);
            }
        }
        self.len = new_len;
    }
}

impl<B: AsRef<[usize]>> BitVec<B> {
    /// Returns an owned copy of the bit vector.
    pub fn to_owned(&self) -> BitVec {
        BitVec {
            bits: self.bits.as_ref().to_owned(),
            len: self.len,
        }
    }
}

impl<B> BitLength for BitVec<B> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }
}

impl<B: AsRef<[usize]>> BitCount for BitVec<B> {
    fn count_ones(&self) -> usize {
        let full_words = self.len() / BITS;
        let residual = self.len() % BITS;
        let bits = self.as_ref();
        let mut num_ones = bits[..full_words]
            .iter()
            .map(|x| x.count_ones() as usize)
            .sum();
        if residual != 0 {
            num_ones += (self.as_ref()[full_words] << (BITS - residual)).count_ones() as usize
        }
        num_ones
    }
}

impl<B: AsRef<[usize]>, C: AsRef<[usize]>> PartialEq<BitVec<C>> for BitVec<B> {
    fn eq(&self, other: &BitVec<C>) -> bool {
        let len = self.len();
        if len != other.len() {
            return false;
        }

        let full_words = len / BITS;
        if self.as_ref()[..full_words] != other.as_ref()[..full_words] {
            return false;
        }

        let residual = len % BITS;

        residual == 0
            || (self.as_ref()[full_words] ^ other.as_ref()[full_words]) << (BITS - residual) == 0
    }
}

impl<B: AsRef<[usize]>> Eq for BitVec<B> {}

impl<B: AsRef<[usize]>> Index<usize> for BitVec<B> {
    type Output = bool;

    fn index(&self, index: usize) -> &Self::Output {
        match self.get(index) {
            false => &false,
            true => &true,
        }
    }
}

impl Extend<bool> for BitVec<Vec<usize>> {
    fn extend<T>(&mut self, i: T)
    where
        T: IntoIterator<Item = bool>,
    {
        for b in i {
            self.push(b);
        }
    }
}

impl FromIterator<bool> for BitVec<Vec<usize>> {
    fn from_iter<T: IntoIterator<Item = bool>>(iter: T) -> Self {
        let mut res = Self::new(0);
        res.extend(iter);
        res
    }
}

impl<'a, B: AsRef<[usize]>> IntoIterator for &'a BitVec<B> {
    type IntoIter = BitIter<'a, B>;
    type Item = bool;

    fn into_iter(self) -> Self::IntoIter {
        BitIter::new(&self.bits, self.len())
    }
}

impl<B: AsRef<[usize]>> fmt::Display for BitVec<B> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[")?;
        for b in self {
            write!(f, "{:b}", b as usize)?;
        }
        write!(f, "]")?;
        Ok(())
    }
}

#[derive(Debug, Clone, MemDbg, MemSize)]
/// A thread-safe bit vector.
///
/// See the [module documentation](mod@crate::bits::bit_vec) for details.
pub struct AtomicBitVec<B = Box<[AtomicUsize]>> {
    bits: B,
    len: usize,
}

impl<B> AtomicBitVec<B> {
    /// Returns the number of bits in the bit vector.
    ///
    /// This method is equivalent to [`BitLength::len`], but it is provided to
    /// reduce ambiguity in method resolution.
    #[inline(always)]
    pub fn len(&self) -> usize {
        BitLength::len(self)
    }

    /// # Safety
    /// `len` must be between 0 (included) and the number of
    /// bits in `bits` (included).
    #[inline(always)]
    pub unsafe fn from_raw_parts(bits: B, len: usize) -> Self {
        Self { bits, len }
    }
    #[inline(always)]
    pub fn into_raw_parts(self) -> (B, usize) {
        (self.bits, self.len)
    }
}

impl AtomicBitVec<Box<[AtomicUsize]>> {
    /// Creates a new atomic bit vector of length `len` initialized to `false`.
    pub fn new(len: usize) -> Self {
        Self::with_value(len, false)
    }

    /// Creates a new atomic bit vector of length `len` initialized to `value`.
    pub fn with_value(len: usize, value: bool) -> Self {
        let n_of_words = len.div_ceil(BITS);
        let extra_bits = (n_of_words * BITS) - len;
        let word_value = if value { !0 } else { 0 };
        let mut bits = (0..n_of_words)
            .map(|_| AtomicUsize::new(word_value))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        if extra_bits > 0 {
            let last_word_value = word_value >> extra_bits;
            bits[n_of_words - 1] = AtomicUsize::new(last_word_value);
        }
        Self { bits, len }
    }
}

impl<B> BitLength for AtomicBitVec<B> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }
}

impl<B: AsRef<[AtomicUsize]>> BitCount for AtomicBitVec<B> {
    fn count_ones(&self) -> usize {
        let full_words = self.len() / BITS;
        let residual = self.len() % BITS;
        let bits = self.as_ref();
        let mut num_ones;
        // Just to be sure, add a fence to ensure that we will see all the final
        // values
        core::sync::atomic::fence(Ordering::SeqCst);
        num_ones = bits[..full_words]
            .iter()
            .map(|x| x.load(Ordering::Relaxed).count_ones() as usize)
            .sum();
        if residual != 0 {
            num_ones += (bits[full_words].load(Ordering::Relaxed) << (BITS - residual)).count_ones()
                as usize
        }
        num_ones
    }
}

impl<B: AsRef<[AtomicUsize]>> Index<usize> for AtomicBitVec<B> {
    type Output = bool;

    /// Shorthand for `get` using [`Ordering::Relaxed`].
    fn index(&self, index: usize) -> &Self::Output {
        match self.get(index, Ordering::Relaxed) {
            false => &false,
            true => &true,
        }
    }
}

impl<'a, B: AsRef<[AtomicUsize]>> IntoIterator for &'a AtomicBitVec<B> {
    type IntoIter = AtomicBitIter<'a, B>;
    type Item = bool;

    fn into_iter(self) -> Self::IntoIter {
        AtomicBitIter::new(&self.bits, self.len())
    }
}

// Conversions

/// This conversion may fail if the alignment of `W` is not the same as
/// that of `W::AtomicType`.
impl<'a, W: IntoAtomic> TryFrom<BitVec<&'a [W]>> for AtomicBitVec<&'a [W::AtomicType]> {
    type Error = CannotCastToAtomicError<W>;
    fn try_from(value: BitVec<&'a [W]>) -> Result<Self, Self::Error> {
        if core::mem::align_of::<W>() != core::mem::align_of::<W::AtomicType>() {
            return Err(CannotCastToAtomicError::default());
        }
        Ok(AtomicBitVec {
            bits: unsafe { core::mem::transmute::<&'a [W], &'a [W::AtomicType]>(value.bits) },
            len: value.len,
        })
    }
}

/// This conversion may fail if the alignment of `W` is not the same as
/// that of `W::AtomicType`.
impl<'a, W: IntoAtomic> TryFrom<BitVec<&'a mut [W]>> for AtomicBitVec<&'a mut [W::AtomicType]> {
    type Error = CannotCastToAtomicError<W>;
    fn try_from(value: BitVec<&'a mut [W]>) -> Result<Self, Self::Error> {
        if core::mem::align_of::<W>() != core::mem::align_of::<W::AtomicType>() {
            return Err(CannotCastToAtomicError::default());
        }
        Ok(AtomicBitVec {
            bits: unsafe {
                core::mem::transmute::<&'a mut [W], &'a mut [W::AtomicType]>(value.bits)
            },
            len: value.len,
        })
    }
}

impl<W: IntoAtomic> From<AtomicBitVec<Box<[W::AtomicType]>>> for BitVec<Vec<W>> {
    fn from(value: AtomicBitVec<Box<[W::AtomicType]>>) -> Self {
        BitVec {
            bits: transmute_vec_from_atomic::<W::AtomicType>(value.bits.into_vec()),
            len: value.len,
        }
    }
}

impl<W: IntoAtomic> From<BitVec<Vec<W>>> for AtomicBitVec<Box<[W::AtomicType]>> {
    fn from(value: BitVec<Vec<W>>) -> Self {
        AtomicBitVec {
            bits: transmute_vec_into_atomic(value.bits).into_boxed_slice(),
            len: value.len,
        }
    }
}

impl<W: IntoAtomic> From<AtomicBitVec<Box<[W::AtomicType]>>> for BitVec<Box<[W]>> {
    fn from(value: AtomicBitVec<Box<[W::AtomicType]>>) -> Self {
        BitVec {
            bits: transmute_boxed_slice_from_atomic::<W::AtomicType>(value.bits),
            len: value.len,
        }
    }
}

impl<W: IntoAtomic + Copy> From<BitVec<Box<[W]>>> for AtomicBitVec<Box<[W::AtomicType]>> {
    fn from(value: BitVec<Box<[W]>>) -> Self {
        AtomicBitVec {
            bits: transmute_boxed_slice_into_atomic::<W>(value.bits),
            len: value.len,
        }
    }
}

impl<'a, W: IntoAtomic> From<AtomicBitVec<&'a [W::AtomicType]>> for BitVec<&'a [W]> {
    fn from(value: AtomicBitVec<&'a [W::AtomicType]>) -> Self {
        BitVec {
            bits: unsafe { core::mem::transmute::<&'a [W::AtomicType], &'a [W]>(value.bits) },
            len: value.len,
        }
    }
}

impl<'a, W: IntoAtomic> From<AtomicBitVec<&'a mut [W::AtomicType]>> for BitVec<&'a mut [W]> {
    fn from(value: AtomicBitVec<&'a mut [W::AtomicType]>) -> Self {
        BitVec {
            bits: unsafe {
                core::mem::transmute::<&'a mut [W::AtomicType], &'a mut [W]>(value.bits)
            },
            len: value.len,
        }
    }
}

impl<W> From<BitVec<Vec<W>>> for BitVec<Box<[W]>> {
    fn from(value: BitVec<Vec<W>>) -> Self {
        BitVec {
            bits: value.bits.into_boxed_slice(),
            len: value.len,
        }
    }
}

impl<W> From<BitVec<Box<[W]>>> for BitVec<Vec<W>> {
    fn from(value: BitVec<Box<[W]>>) -> Self {
        BitVec {
            bits: value.bits.into_vec(),
            len: value.len,
        }
    }
}

impl<W, B: AsRef<[W]>> AsRef<[W]> for BitVec<B> {
    #[inline(always)]
    fn as_ref(&self) -> &[W] {
        self.bits.as_ref()
    }
}

impl<W, B: AsMut<[W]>> AsMut<[W]> for BitVec<B> {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut [W] {
        self.bits.as_mut()
    }
}

impl<W, B: AsRef<[W]>> AsRef<[W]> for AtomicBitVec<B> {
    #[inline(always)]
    fn as_ref(&self) -> &[W] {
        self.bits.as_ref()
    }
}

impl<B: AsRef<[usize]>> RankHinted<64> for BitVec<B> {
    #[inline(always)]
    unsafe fn rank_hinted(&self, pos: usize, hint_pos: usize, hint_rank: usize) -> usize {
        let bits = self.as_ref();
        let mut rank = hint_rank;
        let mut hint_pos = hint_pos;

        debug_assert!(
            hint_pos < bits.len(),
            "hint_pos: {}, len: {}",
            hint_pos,
            bits.len()
        );

        while (hint_pos + 1) * 64 <= pos {
            rank += unsafe { bits.get_unchecked(hint_pos).count_ones() } as usize;
            hint_pos += 1;
        }

        rank + (unsafe { bits.get_unchecked(hint_pos) } & ((1 << (pos % 64)) - 1)).count_ones()
            as usize
    }
}

impl<B: AsRef<[usize]>> SelectHinted for BitVec<B> {
    unsafe fn select_hinted(&self, rank: usize, hint_pos: usize, hint_rank: usize) -> usize {
        let mut word_index = hint_pos / BITS;
        let bit_index = hint_pos % BITS;
        let mut residual = rank - hint_rank;
        let mut word =
            (unsafe { self.as_ref().get_unchecked(word_index) } >> bit_index) << bit_index;
        loop {
            let bit_count = word.count_ones() as usize;
            if residual < bit_count {
                return word_index * BITS + word.select_in_word(residual);
            }
            word_index += 1;
            word = *unsafe { self.as_ref().get_unchecked(word_index) };
            residual -= bit_count;
        }
    }
}

impl<B: AsRef<[usize]>> SelectZeroHinted for BitVec<B> {
    unsafe fn select_zero_hinted(&self, rank: usize, hint_pos: usize, hint_rank: usize) -> usize {
        let mut word_index = hint_pos / BITS;
        let bit_index = hint_pos % BITS;
        let mut residual = rank - hint_rank;
        let mut word =
            (!*unsafe { self.as_ref().get_unchecked(word_index) } >> bit_index) << bit_index;
        loop {
            let bit_count = word.count_ones() as usize;
            if residual < bit_count {
                return word_index * BITS + word.select_in_word(residual);
            }
            word_index += 1;
            word = unsafe { !self.as_ref().get_unchecked(word_index) };
            residual -= bit_count;
        }
    }
}
