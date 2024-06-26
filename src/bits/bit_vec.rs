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
//! These flavors depends on a backend, and presently we provide:
//!
//! - `BitVec<Vec<usize>>`: a mutable, growable and resizable bit vector;
//! - `BitVec<AsRef<[usize]>>`: an immutable bit vector, useful for
//!   [ε-serde](epserde) support;
//! - `BitVec<AsRef<[usize]> + AsMut<[usize]>>`: a mutable (but not resizable)
//!    bit vector;
//! - `AtomicBitVec<AsRef<[AtomicUsize]>>`: a thread-safe, mutable (but not
//!   resizable) bit vector.
//!
//! Note that nothing is assumed about the content of the backend outside the
//! bits of the bit vector. Moreover, the content of the backend outside of
//! the bit vector is never modified by the methods of this class.
//!
//! It is possible to juggle between the three flavors using [`From`]/[`Into`].
//!
//! # Examples
//!
//! ```rust
//! use sux::bit_vec;
//! use sux::traits::{BitCount, BitLength, NumBits, AddNumBits};
//! use sux::bits::{BitVec, AtomicBitVec};
//! use core::sync::atomic::Ordering;
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
//! // If we create an artifically dirty bit vector, everything still works.
//! let ones = [usize::MAX; 2];
//! assert_eq!(unsafe { BitVec::from_raw_parts(ones, 1) }.count_ones(), 1);
//! ```

use common_traits::{IntoAtomic, SelectInWord};
#[allow(unused_imports)] // this is in the std prelude but not in no_std!
use core::borrow::BorrowMut;
use core::fmt;
use epserde::*;
use mem_dbg::*;
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use std::{
    ops::Index,
    sync::atomic::{AtomicUsize, Ordering},
};

use crate::traits::rank_sel::*;

const BITS: usize = usize::BITS as usize;

/// Convenient, [`vec!`](vec!)-like macro to initialize bit vectors.
///
/// # Examples
///
/// ```rust
/// use sux::bit_vec;
/// use sux::traits::BitLength;
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

macro_rules! panic_if_out_of_bounds {
    ($index: expr, $len: expr) => {
        if $index >= $len {
            panic!("Bit index out of bounds: {} >= {}", $index, $len)
        }
    };
}

#[derive(Epserde, Debug, Clone, MemDbg, MemSize)]
/// A bit vector.
pub struct BitVec<B = Vec<usize>> {
    bits: B,
    len: usize,
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
    /// `len` must be between 0 (included) the number of
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

impl<B: AsRef<[usize]>> BitVec<B> {
    pub fn get(&self, index: usize) -> bool {
        panic_if_out_of_bounds!(index, self.len);
        unsafe { self.get_unchecked(index) }
    }

    /// # Safety
    ///
    /// `index` must be between 0 (included) and [`BitVec::len`] (excluded).
    pub unsafe fn get_unchecked(&self, index: usize) -> bool {
        let word_index = index / BITS;
        let word = self.bits.as_ref().get_unchecked(word_index);
        (word >> (index % BITS)) & 1 != 0
    }
}

impl<B: AsRef<[usize]> + AsMut<[usize]>> BitVec<B> {
    pub fn set(&mut self, index: usize, value: bool) {
        panic_if_out_of_bounds!(index, self.len);
        unsafe { self.set_unchecked(index, value) }
    }

    /// # Safety
    ///
    /// `index` must be between 0 (included) and [`BitVec::len`] (excluded).
    #[inline(always)]
    pub unsafe fn set_unchecked(&mut self, index: usize, value: bool) {
        let word_index = index / BITS;
        let bit_index = index % BITS;
        let bits = self.bits.as_mut();
        // TODO: no test?
        // For constant values, this should be inlined with no test.
        if value {
            *bits.get_unchecked_mut(word_index) |= 1 << bit_index;
        } else {
            *bits.get_unchecked_mut(word_index) &= !(1 << bit_index);
        }
    }

    /// Set all bits to the given value.
    ///
    /// If the feature "rayon" is enabled, this method is computed in parallel.
    pub fn fill(&mut self, value: bool) {
        let full_words = self.len() / BITS;
        let residual = self.len % BITS;
        let bits = self.bits.as_mut();
        let word_value = if value { !0 } else { 0 };

        #[cfg(feature = "rayon")]
        {
            bits[..full_words]
                .par_iter_mut()
                .for_each(|x| *x = word_value);
        }

        #[cfg(not(feature = "rayon"))]
        {
            bits[..full_words].iter_mut().for_each(|x| *x = word_value);
        }

        if residual != 0 {
            let mask = (1 << residual) - 1;
            bits[full_words] = (bits[full_words] & !mask) | (word_value & mask);
        }
    }

    /// Flip all bits.
    ///
    /// If the feature "rayon" is enabled, this method is computed in parallel.
    pub fn flip(&mut self) {
        let full_words = self.len() / BITS;
        let residual = self.len % BITS;
        let bits = self.bits.as_mut();

        #[cfg(feature = "rayon")]
        {
            bits[..full_words].par_iter_mut().for_each(|x| *x = !*x);
        }

        #[cfg(not(feature = "rayon"))]
        {
            bits[..full_words].iter_mut().for_each(|x| *x = !*x);
        }

        if residual != 0 {
            let mask = (1 << residual) - 1;
            bits[full_words] = (bits[full_words] & !mask) | (!bits[full_words] & mask);
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
        let n_of_words = (len + BITS - 1) / BITS;
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

    pub fn capacity(&self) -> usize {
        self.bits.capacity() * BITS
    }

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

    pub fn pop(&mut self) -> Option<bool> {
        if self.len == 0 {
            return None;
        }
        self.len -= 1;
        let word_index = self.len / BITS;
        let bit_index = self.len % BITS;
        Some((self.bits[word_index] >> bit_index) & 1 != 0)
    }

    pub fn resize(&mut self, new_len: usize, value: bool) {
        // TODO: rewrite by word
        if new_len > self.len {
            if new_len > self.bits.len() * BITS {
                self.bits.resize((new_len + BITS - 1) / BITS, 0);
            }
            for i in self.len..new_len {
                unsafe {
                    self.set_unchecked(i, value);
                }
            }
        }
        self.len = new_len;
    }
}

impl<B> BitLength for BitVec<B> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }
}

/// If the feature "rayon" is enabled, [`count_ones`](BitCount::count_ones) is
/// computed in parallel.
impl<B: AsRef<[usize]>> BitCount for BitVec<B> {
    fn count_ones(&self) -> usize {
        let full_words = self.len() / BITS;
        let residual = self.len() % BITS;
        let bits = self.bits.as_ref();
        let mut num_ones;

        #[cfg(feature = "rayon")]
        {
            num_ones = bits[..full_words]
                .par_iter()
                .map(|x| x.count_ones() as usize)
                .sum();
        }

        #[cfg(not(feature = "rayon"))]
        {
            num_ones = bits[..full_words]
                .iter()
                .map(|x| x.count_ones() as usize)
                .sum()
        }

        if residual != 0 {
            num_ones += (self.as_ref()[full_words] << (BITS - residual)).count_ones() as usize
        }

        num_ones
    }
}

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
            rank += bits.get_unchecked(hint_pos).count_ones() as usize;
            hint_pos += 1;
        }

        rank + (bits.get_unchecked(hint_pos) & ((1 << (pos % 64)) - 1)).count_ones() as usize
    }
}

impl<B: AsRef<[usize]>> SelectHinted for BitVec<B> {
    unsafe fn select_hinted(&self, rank: usize, hint_pos: usize, hint_rank: usize) -> usize {
        let mut word_index = hint_pos / BITS;
        let bit_index = hint_pos % BITS;
        let mut residual = rank - hint_rank;
        let mut word = (self.as_ref().get_unchecked(word_index) >> bit_index) << bit_index;
        loop {
            let bit_count = word.count_ones() as usize;
            if residual < bit_count {
                return word_index * BITS + word.select_in_word(residual);
            }
            word_index += 1;
            word = *self.as_ref().get_unchecked(word_index);
            residual -= bit_count;
        }
    }
}

impl<B: AsRef<[usize]>> SelectZeroHinted for BitVec<B> {
    unsafe fn select_zero_hinted(&self, rank: usize, hint_pos: usize, hint_rank: usize) -> usize {
        let mut word_index = hint_pos / BITS;
        let bit_index = hint_pos % BITS;
        let mut residual = rank - hint_rank;
        let mut word = (!*self.as_ref().get_unchecked(word_index) >> bit_index) << bit_index;
        loop {
            let bit_count = word.count_ones() as usize;
            if residual < bit_count {
                return word_index * BITS + word.select_in_word(residual);
            }
            word_index += 1;
            word = !self.as_ref().get_unchecked(word_index);
            residual -= bit_count;
        }
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

impl Eq for BitVec<Vec<usize>> {}

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

// An iterator over the bits of this bit vector as booleans.
#[derive(Debug, Clone, MemDbg, MemSize)]
pub struct BitIterator<'a, B> {
    bits: &'a B,
    len: usize,
    next_bit_pos: usize,
}

impl<'a, B: AsRef<[usize]>> IntoIterator for &'a BitVec<B> {
    type IntoIter = BitIterator<'a, B>;
    type Item = bool;

    fn into_iter(self) -> Self::IntoIter {
        BitIterator {
            bits: &self.bits,
            len: self.len,
            next_bit_pos: 0,
        }
    }
}

impl<'a, B: AsRef<[usize]>> Iterator for BitIterator<'a, B> {
    type Item = bool;
    fn next(&mut self) -> Option<bool> {
        if self.next_bit_pos == self.len {
            return None;
        }
        let word_idx = self.next_bit_pos / BITS;
        let bit_idx = self.next_bit_pos % BITS;
        let word = unsafe { *self.bits.as_ref().get_unchecked(word_idx) };
        let bit = (word >> bit_idx) & 1;
        self.next_bit_pos += 1;
        Some(bit != 0)
    }
}

/// An iterator over the positions of the ones in a bit vector.
#[derive(Debug, Clone, MemDbg, MemSize)]
pub struct OnesIterator<'a, B> {
    bits: &'a B,
    len: usize,
    word_idx: usize,
    /// This is a usize because BitVec is currently implemented only for `Vec<usize>` and `&[usize]`.
    word: usize,
}

impl<'a, B: AsRef<[usize]>> OnesIterator<'a, B> {
    pub fn new(bits: &'a B, len: usize) -> Self {
        let word = if bits.as_ref().is_empty() {
            0
        } else {
            unsafe { *bits.as_ref().get_unchecked(0) }
        };
        Self {
            bits,
            len,
            word_idx: 0,
            word,
        }
    }
}

impl<'a, B: AsRef<[usize]>> Iterator for OnesIterator<'a, B> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        // find the next word with ones
        while self.word == 0 {
            self.word_idx += 1;
            if self.word_idx == self.bits.as_ref().len() {
                return None;
            }
            self.word = unsafe { *self.bits.as_ref().get_unchecked(self.word_idx) };
        }
        // find the lowest bit set index in the word
        let bit_idx = self.word.trailing_zeros() as usize;
        // compute the global bit index
        let res = (self.word_idx * BITS) + bit_idx;
        if res >= self.len {
            None
        } else {
            // clear the lowest bit set
            self.word &= self.word - 1;
            Some(res)
        }
    }
}

/// An iterator over the positions of the zeros in a bit vector.
#[derive(Debug, Clone, MemDbg, MemSize)]
pub struct ZerosIterator<'a, B> {
    bits: &'a B,
    len: usize,
    word_idx: usize,
    /// This is a usize because BitVec is currently implemented only for `Vec<usize>` and `&[usize]`.
    word: usize,
}

impl<'a, B: AsRef<[usize]>> ZerosIterator<'a, B> {
    pub fn new(bits: &'a B, len: usize) -> Self {
        let word = if bits.as_ref().is_empty() {
            0
        } else {
            unsafe { !*bits.as_ref().get_unchecked(0) }
        };
        Self {
            bits,
            len,
            word_idx: 0,
            word,
        }
    }
}

impl<'a, B: AsRef<[usize]>> Iterator for ZerosIterator<'a, B> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        // find the next flipped word with zeros
        while self.word == 0 {
            self.word_idx += 1;
            if self.word_idx == self.bits.as_ref().len() {
                return None;
            }
            self.word = unsafe { !*self.bits.as_ref().get_unchecked(self.word_idx) };
        }
        // find the lowest zero bit index in the word
        let bit_idx = self.word.trailing_zeros() as usize;
        // compute the global bit index
        let res = (self.word_idx * BITS) + bit_idx;
        if res >= self.len {
            None
        } else {
            // clear the lowest bit set
            self.word &= self.word - 1;
            Some(res)
        }
    }
}

impl<B: AsRef<[usize]>> BitVec<B> {
    // Returns an iterator over the bits of this bit vector.
    #[inline(always)]
    pub fn iter(&self) -> BitIterator<B> {
        self.into_iter()
    }

    // Returns an iterator over the positions of the ones in this bit vector.
    pub fn iter_ones(&self) -> OnesIterator<B> {
        OnesIterator::new(&self.bits, self.len)
    }

    // Returns an iterator over the positions of the zeros in this bit vector.
    pub fn iter_zeros(&self) -> ZerosIterator<B> {
        ZerosIterator::new(&self.bits, self.len)
    }
}

#[derive(Debug, Clone, MemDbg, MemSize)]
/// A thread-safe bit vector.
pub struct AtomicBitVec<B = Vec<AtomicUsize>> {
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
    /// `len` must be between 0 (included) the number of
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

impl<B: AsRef<[AtomicUsize]>> AtomicBitVec<B> {
    pub fn get(&self, index: usize, ordering: Ordering) -> bool {
        panic_if_out_of_bounds!(index, self.len);
        unsafe { self.get_unchecked(index, ordering) }
    }

    pub fn set(&self, index: usize, value: bool, ordering: Ordering) {
        panic_if_out_of_bounds!(index, self.len);
        unsafe { self.set_unchecked(index, value, ordering) }
    }

    pub fn swap(&self, index: usize, value: bool, ordering: Ordering) -> bool {
        panic_if_out_of_bounds!(index, self.len);
        unsafe { self.swap_unchecked(index, value, ordering) }
    }

    unsafe fn get_unchecked(&self, index: usize, ordering: Ordering) -> bool {
        let word_index = index / BITS;
        let bits = self.bits.as_ref();
        let word = bits.get_unchecked(word_index).load(ordering);
        (word >> (index % BITS)) & 1 != 0
    }
    #[inline(always)]
    unsafe fn set_unchecked(&self, index: usize, value: bool, ordering: Ordering) {
        let word_index = index / BITS;
        let bit_index = index % BITS;
        let bits = self.bits.as_ref();

        // For constant values, this should be inlined with no test.
        if value {
            bits.get_unchecked(word_index)
                .fetch_or(1 << bit_index, ordering);
        } else {
            bits.get_unchecked(word_index)
                .fetch_and(!(1 << bit_index), ordering);
        }
    }

    #[inline(always)]
    unsafe fn swap_unchecked(&self, index: usize, value: bool, ordering: Ordering) -> bool {
        let word_index = index / BITS;
        let bit_index = index % BITS;
        let bits = self.bits.as_ref();

        let old_word = if value {
            bits.get_unchecked(word_index)
                .fetch_or(1 << bit_index, ordering)
        } else {
            bits.get_unchecked(word_index)
                .fetch_and(!(1 << bit_index), ordering)
        };

        (old_word >> (bit_index)) & 1 != 0
    }

    /// Set all bits to the given value.
    ///
    /// If the feature "rayon" is enabled, this method is computed in parallel.
    pub fn fill(&mut self, value: bool, ordering: Ordering) {
        let full_words = self.len() / BITS;
        let residual = self.len % BITS;
        let bits = self.bits.as_ref();
        let word_value = if value { !0 } else { 0 };

        // Just to be sure, add a fence to ensure that we will see all the final
        // values
        core::sync::atomic::fence(Ordering::SeqCst);
        #[cfg(feature = "rayon")]
        {
            bits[..full_words]
                .par_iter()
                .for_each(|x| x.store(word_value, ordering));
        }

        #[cfg(not(feature = "rayon"))]
        {
            bits[..full_words]
                .iter()
                .for_each(|x| x.store(word_value, ordering));
        }

        if residual != 0 {
            let mask = (1 << residual) - 1;
            bits[full_words].store(
                (bits[full_words].load(ordering) & !mask) | (word_value & mask),
                ordering,
            );
        }
    }

    /// Flip all bits.
    ///
    /// If the feature "rayon" is enabled, this method is computed in parallel.
    pub fn flip(&mut self, ordering: Ordering) {
        let full_words = self.len() / BITS;
        let residual = self.len % BITS;
        let bits = self.bits.as_ref();

        // Just to be sure, add a fence to ensure that we will see all the final
        // values
        core::sync::atomic::fence(Ordering::SeqCst);
        #[cfg(feature = "rayon")]
        {
            bits[..full_words]
                .par_iter()
                .for_each(|x| _ = x.fetch_xor(!0, ordering));
        }

        #[cfg(not(feature = "rayon"))]
        {
            bits[..full_words]
                .iter()
                .for_each(|x| _ = x.fetch_xor(!0, ordering));
        }

        if residual != 0 {
            let mask = (1 << residual) - 1;
            let last_word = bits[full_words].load(ordering);
            bits[full_words].store((last_word & !mask) | (!last_word & mask), ordering);
        }
    }
}

impl AtomicBitVec<Vec<AtomicUsize>> {
    /// Creates a new atomic bit vector of length `len` initialized to `false`.
    pub fn new(len: usize) -> Self {
        Self::with_value(len, false)
    }

    /// Creates a new atomic bit vector of length `len` initialized to `value`.
    pub fn with_value(len: usize, value: bool) -> Self {
        let n_of_words = (len + BITS - 1) / BITS;
        let extra_bits = (n_of_words * BITS) - len;
        let word_value = if value { !0 } else { 0 };
        let mut bits = (0..n_of_words)
            .map(|_| AtomicUsize::new(word_value))
            .collect::<Vec<_>>();
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

impl<B: AsRef<[AtomicUsize]>> Index<usize> for AtomicBitVec<B> {
    type Output = bool;

    /// Shorthand for [`Self::get`] using [`Ordering::Relaxed`].
    fn index(&self, index: usize) -> &Self::Output {
        match self.get(index, Ordering::Relaxed) {
            false => &false,
            true => &true,
        }
    }
}

/// If the feature "rayon" is enabled, [`count_ones`](BitCount::count_ones) is
/// computed in parallel.
impl<B: AsRef<[AtomicUsize]>> BitCount for AtomicBitVec<B> {
    fn count_ones(&self) -> usize {
        let full_words = self.len() / BITS;
        let residual = self.len() % BITS;
        let bits = self.bits.as_ref();
        let mut num_ones;

        // Just to be sure, add a fence to ensure that we will see all the final
        // values
        core::sync::atomic::fence(Ordering::SeqCst);
        #[cfg(feature = "rayon")]
        {
            num_ones = bits[..full_words]
                .par_iter()
                .map(|x| x.load(Ordering::Relaxed).count_ones() as usize)
                .sum();
        }

        #[cfg(not(feature = "rayon"))]
        {
            num_ones = bits[..full_words]
                .iter()
                .map(|x| x.load(Ordering::Relaxed).count_ones() as usize)
                .sum();
        }

        if residual != 0 {
            num_ones += (bits[full_words].load(Ordering::Relaxed) << (BITS - residual)).count_ones()
                as usize
        }

        num_ones
    }
}

// Conversions

impl<W: IntoAtomic> From<BitVec<Vec<W>>> for AtomicBitVec<Vec<W::AtomicType>> {
    fn from(value: BitVec<Vec<W>>) -> Self {
        AtomicBitVec {
            bits: unsafe { core::mem::transmute::<Vec<W>, Vec<W::AtomicType>>(value.bits) },
            len: value.len,
        }
    }
}

impl<'a, W: IntoAtomic> From<BitVec<&'a [W]>> for AtomicBitVec<&'a [W::AtomicType]> {
    fn from(value: BitVec<&'a [W]>) -> Self {
        AtomicBitVec {
            bits: unsafe { core::mem::transmute::<&'a [W], &'a [W::AtomicType]>(value.bits) },
            len: value.len,
        }
    }
}

impl<'a, W: IntoAtomic> From<BitVec<&'a mut [W]>> for AtomicBitVec<&'a mut [W::AtomicType]> {
    fn from(value: BitVec<&'a mut [W]>) -> Self {
        AtomicBitVec {
            bits: unsafe {
                core::mem::transmute::<&'a mut [W], &'a mut [W::AtomicType]>(value.bits)
            },
            len: value.len,
        }
    }
}

impl<W: IntoAtomic> From<AtomicBitVec<Vec<W::AtomicType>>> for BitVec<Vec<W>> {
    fn from(value: AtomicBitVec<Vec<W::AtomicType>>) -> Self {
        BitVec {
            bits: unsafe { core::mem::transmute::<Vec<W::AtomicType>, Vec<W>>(value.bits) },
            len: value.len,
        }
    }
}

impl<W: IntoAtomic> From<AtomicBitVec<Box<[W::AtomicType]>>> for BitVec<Box<[W]>> {
    fn from(value: AtomicBitVec<Box<[W::AtomicType]>>) -> Self {
        BitVec {
            bits: unsafe { core::mem::transmute::<Box<[W::AtomicType]>, Box<[W]>>(value.bits) },
            len: value.len,
        }
    }
}

impl<W: IntoAtomic> From<BitVec<Box<[W]>>> for AtomicBitVec<Box<[W::AtomicType]>> {
    fn from(value: BitVec<Box<[W]>>) -> Self {
        AtomicBitVec {
            bits: unsafe { core::mem::transmute::<Box<[W]>, Box<[W::AtomicType]>>(value.bits) },
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
// An iterator over the bits of this atomic bit vector as booleans.
#[derive(Debug, MemDbg, MemSize)]
pub struct AtomicBitIterator<'a, B> {
    bits: &'a mut B,
    len: usize,
    next_bit_pos: usize,
}

// We implement [`IntoIterator`] for a mutable reference so no
// outstanding references are allowed while iterating.
impl<'a, B: AsRef<[AtomicUsize]>> IntoIterator for &'a mut AtomicBitVec<B> {
    type IntoIter = AtomicBitIterator<'a, B>;
    type Item = bool;

    fn into_iter(self) -> Self::IntoIter {
        AtomicBitIterator {
            bits: &mut self.bits,
            len: self.len,
            next_bit_pos: 0,
        }
    }
}

impl<'a, B: AsRef<[AtomicUsize]>> Iterator for AtomicBitIterator<'a, B> {
    type Item = bool;
    fn next(&mut self) -> Option<bool> {
        if self.next_bit_pos == self.len {
            return None;
        }
        let word_idx = self.next_bit_pos / BITS;
        let bit_idx = self.next_bit_pos % BITS;
        let word = unsafe {
            self.bits
                .as_ref()
                .get_unchecked(word_idx)
                .load(Ordering::Relaxed)
        };
        let bit = (word >> bit_idx) & 1;
        self.next_bit_pos += 1;
        Some(bit != 0)
    }
}

impl<B: AsRef<[AtomicUsize]>> AtomicBitVec<B> {
    // Returns an iterator over the bits of this bit vector.
    //
    // Note that this method takes a mutable reference to the bit vector,
    // so no outstanding references are allowed while iterating.
    #[inline(always)]
    pub fn iter(&mut self) -> AtomicBitIterator<B> {
        self.into_iter()
    }
}
