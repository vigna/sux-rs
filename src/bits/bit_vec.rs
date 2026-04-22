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
//! [`BitVecOps`], [`BitVecOpsMut`], [`BitVecValueOps`], and
//! [`AtomicBitVecOps`], which must be pulled in scope as needed. There are also
//! operations that are specific to certain implementations, such as [`push`].
//!
//! These flavors depend on a backend with a word type `W`, and presently we
//! provide:
//!
//! - `BitVec<Vec<W>>`: a mutable, growable and resizable bit vector;
//! - `BitVec<AsRef<[W]>>`: an immutable bit vector, useful for
//!   [ε-serde] support;
//! - `BitVec<AsRef<[W]> + AsMut<[W]>>`: a mutable (but
//!   not resizable) bit vector;
//! - `AtomicBitVec<AsRef<[Atomic<W>]>>`: a thread-safe, mutable (but
//!   not resizable) bit vector.
//!
//! Note that nothing is assumed about the content of the backend outside the
//! bits of the bit vector. Moreover, the content of the backend outside of
//! the bit vector is never modified by the methods of this structure.
//!
//! It is possible to juggle between all flavors using [`From`]/[`Into`], and
//! with [`TryFrom`]/[`TryInto`] when going [from a non-atomic to an atomic bit vector].
//!
//! # Type annotations
//!
//! Both [`BitVec`] and [`AtomicBitVec`] have default type parameters for
//! their backends. However, Rust does not apply struct default type
//! parameters in expression position, so constructor calls like
//! `BitVec::new(n)` or `AtomicBitVec::new(n)` leave the backend type
//! unconstrained.
//!
//! The fix is to annotate the binding with the bare type alias, which
//! *does* apply defaults:
//!
//! ```rust
//! # use sux::prelude::*;
//! let mut b: BitVec = BitVec::new(10);     // OK: B = Vec<usize>
//! let a: AtomicBitVec = AtomicBitVec::new(10); // OK: B = Box<[Atomic<usize>]>
//! ```
//!
//! The [`bit_vec!`] macro and [`FromIterator`] / [`Extend`] do not need
//! annotations because the word type is determined by the output context.
//!
//! # Slice-by-value support
//!
//! [`BitVec`] implement the [`BitFieldSlice`]/[`BitFieldSliceMut`] traits as a
//! bit-field slice of width one. Note that the methods [`get_value`] and
//! [`get_value_unchecked`] of [`SliceByValue`] conflicts with those of
//! [`BitVecValueOps`], so if you absolutely need to pull in both traits you
//! have to disambiguate the method calls. Moreover, as part of [`SliceByValueMut`]
//! you can also obtain [mutable chunks] from a bit vector, provided they are
//! aligned enough.
//!
//! [chunks]: SliceByValueMut::try_chunks_mut
//! [`get_value`]: SliceByValue::get_value
//! [`get_value_unchecked`]: SliceByValue::get_value_unchecked
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
//! let mut b: BitVec = BitVec::new(0);
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
//! assert_eq!(unsafe { BitVec::from_raw_parts(ones.as_slice(), 1) }.count_ones(), 1);
//! ```
//!
//! [from a non-atomic to an atomic bit vector]: BitVec#impl-TryFrom%3CBitVec%3C%26%5BW%5D%3E%3E-for-AtomicBitVec%3C%26%5B%3CW+as+AtomicPrimitive%3E%3A%3AAtomic%5D%3E
//! [ε-serde]: https://crates.io/crates/epserde
//! [`push`]: BitVec::push
//! [`bit_vec!`]: macro@crate::bits::bit_vec

use crate::ambassador_impl_Index;
use crate::traits::ambassador_impl_Backend;
use crate::traits::ambassador_impl_BitLength;
use crate::traits::{
    AtomicBitIter, AtomicBitVecOps, Backend, BitFieldSlice, BitFieldSliceMut, BitIter, BitVecOps,
    BitVecOpsMut, BitVecValueOps, BitWidth, Word,
};
use crate::utils::SelectInWord;
use crate::{
    traits::{bit_vec_ops::BitLength, rank_sel::*},
    utils::{
        CannotCastToAtomicError, transmute_boxed_slice_from_atomic,
        transmute_boxed_slice_into_atomic, transmute_vec_from_atomic, transmute_vec_into_atomic,
    },
};
use ambassador::Delegate;
use atomic_primitive::{Atomic, AtomicPrimitive, PrimitiveAtomic, PrimitiveAtomicUnsigned};
#[allow(unused_imports)] // this is in the std prelude but not in no_std!
use core::borrow::BorrowMut;
use core::fmt;
use mem_dbg::*;
use num_primitive::PrimitiveInteger;
use std::iter::FusedIterator;
use std::{ops::Index, sync::atomic::Ordering};
use value_traits::slices::{SliceByValue, SliceByValueMut};

/// A bit vector.
///
/// Instances can be created using [`new`], [`with_value`], with the
/// convenience macro [`bit_vec!`], or with a [`FromIterator` implementation].
///
/// See the [module documentation] for more details.
///
/// [`FromIterator` implementation]: #impl-FromIterator<bool>-for-BitVec
/// [`new`]: BitVec::new
/// [`with_value`]: BitVec::with_value
/// [`bit_vec!`]: macro@crate::bits::bit_vec
/// [module documentation]: mod@crate::bits::bit_vec
#[derive(Debug, Clone, MemSize, MemDbg, Delegate)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[delegate(crate::traits::Backend, target = "bits")]
pub struct BitVec<B = Vec<usize>> {
    bits: B,
    len: usize,
}

/// Convenient, [`vec!`]-like macro to initialize bit vectors.
///
/// [`vec!`]: vec!
///
/// By default, the underlying storage is `Vec<usize>`. An explicit word type
/// `W` can be selected by prepending `W:` to any form, producing a
/// [`BitVec<Vec<W>>`]; this is useful, for example, to obtain a reproducible
/// layout across platforms with different pointer widths.
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
///   where each `bᵢ` can be any expression that evaluates to a boolean or
///   integer (0 for `false`, non-zero for `true`).
///
/// - `bit_vec![W]`, `bit_vec![W: false; n]`, `bit_vec![W: 0; n]`,
///   `bit_vec![W: true; n]`, `bit_vec![W: 1; n]`, and
///   `bit_vec![W: b₀, b₁, b₂, …]` are the same as the forms above, but
///   backed by `Vec<W>` instead of `Vec<usize>`.
///
/// # Examples
///
/// ```rust
/// # #[cfg(target_pointer_width = "64")]
/// # {
/// # use sux::prelude::*;
/// # use sux::traits::BitVecOps;
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
///
/// // With explicit word type (useful for cross-platform code)
/// let b = bit_vec![u32: 0, 1, 0, 1];
/// assert_eq!(b.len(), 4);
/// let b = bit_vec![u32: false; 10];
/// assert_eq!(b.len(), 10);
/// # }
/// ```
#[macro_export]
macro_rules! bit_vec {
    // Arms with explicit word type (colon separator)
    ($W:ty) => {
        $crate::bits::BitVec::<Vec<$W>>::new(0)
    };
    ($W:ty: false; $n:expr) => {
        $crate::bits::BitVec::<Vec<$W>>::new($n)
    };
    ($W:ty: 0; $n:expr) => {
        $crate::bits::BitVec::<Vec<$W>>::new($n)
    };
    ($W:ty: true; $n:expr) => {
        {
            $crate::bits::BitVec::<Vec<$W>>::with_value($n, true)
        }
    };
    ($W:ty: 1; $n:expr) => {
        {
            $crate::bits::BitVec::<Vec<$W>>::with_value($n, true)
        }
    };
    ($W:ty: $($x:expr),+ $(,)?) => {
        {
            let mut b = $crate::bits::BitVec::<Vec<$W>>::with_capacity([$($x),+].len());
            $( b.push($x != 0); )*
            b
        }
    };
    // Default arms (usize backing)
    () => {
        $crate::bits::BitVec::<Vec<usize>>::new(0)
    };
    (false; $n:expr) => {
        $crate::bits::BitVec::<Vec<usize>>::new($n)
    };
    (0; $n:expr) => {
        $crate::bits::BitVec::<Vec<usize>>::new($n)
    };
    (true; $n:expr) => {
        {
            $crate::bits::BitVec::<Vec<usize>>::with_value($n, true)
        }
    };
    (1; $n:expr) => {
        {
            $crate::bits::BitVec::<Vec<usize>>::with_value($n, true)
        }
    };
    ($($x:expr),+ $(,)?) => {
        {
            let mut b = $crate::bits::BitVec::<Vec<usize>>::with_capacity([$($x),+].len());
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
    pub const fn len(&self) -> usize {
        self.len
    }

    /// # Safety
    /// `len` must be between 0 (included) and the number of
    /// bits in `bits` (included).
    #[inline(always)]
    #[must_use]
    pub const unsafe fn from_raw_parts(bits: B, len: usize) -> Self {
        Self { bits, len }
    }

    /// Returns the backend and the length in bits, consuming this bit vector.
    #[inline(always)]
    pub fn into_raw_parts(self) -> (B, usize) {
        (self.bits, self.len)
    }

    /// Replaces the backend by applying a function, consuming this bit vector.
    ///
    /// # Safety
    /// The caller must ensure that the length is compatible with the new
    /// backend.
    #[inline(always)]
    pub unsafe fn map<B2>(self, f: impl FnOnce(B) -> B2) -> BitVec<B2> {
        BitVec {
            bits: f(self.bits),
            len: self.len,
        }
    }
}

impl<W: Word> BitVec<Vec<W>> {
    /// Creates a new bit vector of length `len` initialized to `false`.
    #[must_use]
    pub fn new(len: usize) -> Self {
        Self::with_value(len, false)
    }

    /// Creates a new bit vector of length `len` initialized to `value`.
    #[must_use]
    pub fn with_value(len: usize, value: bool) -> Self {
        let bits_per_word = W::BITS as usize;
        let n_of_words = len.div_ceil(bits_per_word);
        let extra_bits = (n_of_words * bits_per_word) - len;
        let word_value = if value { !W::ZERO } else { W::ZERO };
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
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        let bits_per_word = W::BITS as usize;
        let n_of_words = capacity.div_ceil(bits_per_word);
        Self {
            bits: Vec::with_capacity(n_of_words),
            len: 0,
        }
    }

    /// Returns the current capacity of this bit vector.
    pub fn capacity(&self) -> usize {
        self.bits.capacity() * W::BITS as usize
    }

    /// Appends a bit to the end of this bit vector.
    pub fn push(&mut self, b: bool) {
        let bits_per_word = W::BITS as usize;
        if self.bits.len() * bits_per_word == self.len {
            self.bits.push(W::ZERO);
        }
        let word_index = self.len / bits_per_word;
        let bit_index = self.len % bits_per_word;
        // Clear bit
        self.bits[word_index] &= !(W::ONE << bit_index);
        // Set bit
        if b {
            self.bits[word_index] |= W::ONE << bit_index;
        }
        self.len += 1;
    }

    /// Appends the lower `width` bits of `value` to the end of this bit
    /// vector.
    ///
    /// # Panics
    ///
    /// Panics if `width` > `W::BITS`.
    pub fn append_value(&mut self, value: W, width: usize) {
        assert!(
            width <= W::BITS as usize,
            "width {} must be at most W::BITS ({})",
            width,
            W::BITS
        );
        if width == 0 {
            return;
        }
        let bits_per_word = W::BITS as usize;
        let l = bits_per_word - width;
        let value = (value << l) >> l;
        let new_len = self.len + width;
        let needed_words = new_len.div_ceil(bits_per_word);
        // Grow the backing storage if necessary.
        self.bits.resize(needed_words, W::ZERO);

        let word_idx = self.len / bits_per_word;
        let bit_idx = self.len % bits_per_word;

        self.bits[word_idx] |= value << bit_idx;
        if bit_idx + width > bits_per_word {
            self.bits[word_idx + 1] = value.wrapping_shr(bit_idx.wrapping_neg() as u32);
        }
        self.len = new_len;
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

    /// Reserves capacity for at least `additional` more bits to be appended.
    ///
    /// After calling `reserve`, capacity will be greater than or equal to
    /// `self.len() + additional`. The allocator may reserve more space to
    /// speculatively avoid frequent reallocations. Does nothing if the
    /// capacity is already sufficient.
    pub fn reserve(&mut self, additional: usize) {
        let needed_words = (self.len + additional).div_ceil(W::BITS as usize);
        self.bits
            .reserve(needed_words.saturating_sub(self.bits.len()));
    }

    /// Reserves the minimum capacity for at least `additional` more bits to
    /// be appended.
    ///
    /// After calling `reserve_exact`, capacity will be greater than or equal
    /// to `self.len() + additional`. Does nothing if the capacity is already
    /// sufficient.
    ///
    /// Note that the allocator may give the collection more space than it
    /// requests. Therefore, capacity cannot be relied upon to be precisely
    /// minimal.
    pub fn reserve_exact(&mut self, additional: usize) {
        let needed_words = (self.len + additional).div_ceil(W::BITS as usize);
        self.bits
            .reserve_exact(needed_words.saturating_sub(self.bits.len()));
    }

    /// Appends the bits of `other` to the end of this bit vector.
    ///
    /// Unlike [`Vec::append`], `other` is not drained: its contents are
    /// copied into `self`.
    pub fn append<B2: AsRef<[W]>>(&mut self, other: &BitVec<B2>) {
        let other_len = other.len;
        if other_len == 0 {
            return;
        }

        let bpw = W::BITS as usize;
        let offset = self.len % bpw;
        let src: &[W] = other.bits.as_ref();
        let src_words = other_len.div_ceil(bpw);
        let new_total = self.len + other_len;
        let new_word_count = new_total.div_ceil(bpw);

        if offset == 0 {
            self.bits.extend_from_slice(&src[..src_words]);
        } else {
            self.bits.reserve(new_word_count - self.bits.len());

            let last_idx = self.bits.len() - 1;
            self.bits[last_idx] |= src[0] << offset;

            let shift_right = bpw - offset;
            for i in 1..src_words {
                self.bits
                    .push((src[i - 1] >> shift_right) | (src[i] << offset));
            }

            if new_word_count > self.bits.len() {
                self.bits.push(src[src_words - 1] >> shift_right);
            }
        }

        self.len = new_total;
    }

    /// Resizes the bit vector in place, extending it with `value` if it is
    /// necessary.
    pub fn resize(&mut self, new_len: usize, value: bool) {
        let bits_per_word = W::BITS as usize;
        if new_len > self.len {
            let old_len = self.len;
            let old_word = old_len / bits_per_word;
            let old_bit = old_len % bits_per_word;
            let word_value = if value { !W::ZERO } else { W::ZERO };

            self.bits
                .resize(new_len.div_ceil(bits_per_word), word_value);

            // Handle the partial word at old_len, then fill all
            // remaining words (which may contain stale data from
            // previous truncations).
            if old_bit != 0 {
                let mask = !W::ZERO << old_bit;
                self.bits[old_word] = (self.bits[old_word] & !mask) | (word_value & mask);
                self.bits[old_word + 1..].fill(word_value);
            } else {
                self.bits[old_word..].fill(word_value);
            }
        }
        self.len = new_len;
    }

    /// Ensures a padding word is present at the end and converts the
    /// backend to `Box<[W]>`.
    ///
    /// The extra word ensures that unaligned reads of `size_of::<W>()`
    /// bytes starting at any byte offset within the data never exceed the
    /// allocation. If the allocation already has more words than needed
    /// for the data, no word is added.
    pub fn into_padded(mut self) -> BitVec<Box<[W]>> {
        let needed = self.len.div_ceil(W::BITS as usize);
        if self.bits.len() <= needed {
            self.bits.push(W::ZERO);
        }
        unsafe { BitVec::from_raw_parts(self.bits.into_boxed_slice(), self.len) }
    }

    #[deprecated(note = "Use `BitVec<Box<[W]>>::new_padded` instead")]
    pub fn new_padded(len: usize) -> BitVec<Box<[W]>> {
        let n_of_words = len.div_ceil(W::BITS as usize);
        unsafe { BitVec::from_raw_parts(vec![W::ZERO; n_of_words + 1].into_boxed_slice(), len) }
    }
}

impl<W: Word> BitVec<Box<[W]>> {
    /// Creates a new bit vector of length `len` initialized to `false`,
    /// with a padding word at the end for safe unaligned reads.
    ///
    /// This constructor is useful for structures implementing
    /// [`TryIntoUnaligned`] that want to avoid reallocations.
    ///
    /// [`TryIntoUnaligned`]: crate::traits::TryIntoUnaligned
    pub fn new_padded(len: usize) -> BitVec<Box<[W]>> {
        let n_of_words = len.div_ceil(W::BITS as usize);
        unsafe { BitVec::from_raw_parts(vec![W::ZERO; n_of_words + 1].into_boxed_slice(), len) }
    }
}

impl<W: Word> Extend<bool> for BitVec<Vec<W>> {
    fn extend<T: IntoIterator<Item = bool>>(&mut self, i: T) {
        for b in i {
            self.push(b);
        }
    }
}

impl<W: Word> FromIterator<bool> for BitVec<Vec<W>> {
    fn from_iter<T: IntoIterator<Item = bool>>(iter: T) -> Self {
        let mut res = Self::new(0);
        res.extend(iter);
        res
    }
}

impl<B: ToOwned> BitVec<B> {
    /// Returns a copy of this bit vector with an owned backend.
    pub fn to_owned(&self) -> BitVec<<B as ToOwned>::Owned> {
        BitVec {
            bits: self.bits.to_owned(),
            len: self.len,
        }
    }
}

impl<B: Backend<Word: Word> + AsRef<[B::Word]>> BitVecValueOps<B::Word> for BitVec<B> {
    fn get_value(&self, pos: usize, width: usize) -> B::Word {
        assert!(
            width <= B::Word::BITS as usize,
            "width {} must be at most W::BITS ({})",
            width,
            B::Word::BITS
        );
        assert!(
            pos + width <= self.len,
            "bit range {}..{} out of bounds for length {}",
            pos,
            pos + width,
            self.len
        );
        // Disambiguate: `SliceByValue::get_value_unchecked` now also
        // exists for `BitVec`, but with a different signature.
        unsafe { <Self as BitVecValueOps<B::Word>>::get_value_unchecked(self, pos, width) }
    }

    #[inline]
    unsafe fn get_value_unchecked(&self, pos: usize, width: usize) -> B::Word {
        let bits = B::Word::BITS as usize;
        let word_index = pos / bits;
        let bit_index = pos % bits;
        let l = bits - width;
        let data = self.bits.as_ref();

        if width == 0 {
            return B::Word::ZERO;
        }

        unsafe {
            if bit_index <= l {
                (*data.get_unchecked(word_index) << (l - bit_index)) >> l
            } else {
                (*data.get_unchecked(word_index) >> bit_index)
                    | ((*data.get_unchecked(word_index + 1))
                        .wrapping_shl(l.wrapping_sub(bit_index) as u32)
                        >> l)
            }
        }
    }
}

impl<B> BitLength for BitVec<B> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }
}

impl<B: Backend<Word: Word> + AsRef<[B::Word]>> BitCount for BitVec<B> {
    fn count_ones(&self) -> usize {
        let bits_per_word = B::Word::BITS as usize;
        let full_words = self.len() / bits_per_word;
        let residual = self.len() % bits_per_word;
        let bits: &[B::Word] = self.as_ref();
        let mut num_ones: usize = bits[..full_words]
            .iter()
            .map(|x| x.count_ones() as usize)
            .sum();
        if residual != 0 {
            num_ones += (bits[full_words] << (bits_per_word - residual)).count_ones() as usize
        }
        num_ones
    }
}

impl<B: Backend<Word: Word> + AsRef<[B::Word]>, C: Backend<Word = B::Word> + AsRef<[B::Word]>>
    PartialEq<BitVec<C>> for BitVec<B>
{
    fn eq(&self, other: &BitVec<C>) -> bool {
        let len = self.len();
        if len != other.len() {
            return false;
        }

        let word_bits = B::Word::BITS as usize;
        let full_words = len / word_bits;
        if self.as_ref()[..full_words] != other.as_ref()[..full_words] {
            return false;
        }

        let residual = len % word_bits;
        residual == 0
            || (self.as_ref()[full_words] ^ other.as_ref()[full_words]) << (word_bits - residual)
                == B::Word::ZERO
    }
}

impl<B: Backend<Word: Word> + AsRef<[B::Word]>> Eq for BitVec<B> {}

impl<B: Backend<Word: Word> + AsRef<[B::Word]>> std::hash::Hash for BitVec<B> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let len = self.len();
        len.hash(state);
        let word_bits = B::Word::BITS as usize;
        let full_words = len / word_bits;
        self.as_ref()[..full_words].hash(state);
        let residual = len % word_bits;
        if residual != 0 {
            // Mask off the padding bits before hashing the last partial word.
            (self.as_ref()[full_words] << (word_bits - residual)).hash(state);
        }
    }
}

impl<B: Backend<Word: Word> + AsRef<[B::Word]>> Index<usize> for BitVec<B> {
    type Output = bool;

    fn index(&self, index: usize) -> &Self::Output {
        match BitVecOps::<B::Word>::get(self, index) {
            false => &false,
            true => &true,
        }
    }
}

impl<'a, B: Backend<Word: Word> + AsRef<[B::Word]>> IntoIterator for &'a BitVec<B> {
    type IntoIter = BitIter<'a, B::Word, B>;
    type Item = bool;

    fn into_iter(self) -> Self::IntoIter {
        BitIter::new(&self.bits, self.len())
    }
}

impl<B: Backend<Word: Word> + AsRef<[B::Word]>> fmt::Display for BitVec<B> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[")?;
        for b in self {
            write!(f, "{:b}", b as usize)?;
        }
        write!(f, "]")?;
        Ok(())
    }
}

/// An iterator over contiguous mutable chunks of a [`BitVec`], yielding
/// [`BitVec<&mut [W]>`] views.
///
/// This struct is created by [`BitVec`]'s [`try_chunks_mut`]
/// implementation. When the vector length is not evenly divided by the
/// chunk size, the last chunk will be shorter.
///
/// [`try_chunks_mut`]: SliceByValueMut::try_chunks_mut
pub struct BitVecChunksMut<'a, W: Word> {
    remaining: usize,
    chunk_size: usize,
    iter: std::slice::ChunksMut<'a, W>,
}

impl<'a, W: Word> Iterator for BitVecChunksMut<'a, W> {
    type Item = BitVec<&'a mut [W]>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|chunk| {
            let size = Ord::min(self.chunk_size, self.remaining);
            // SAFETY: `size` is bounded by the original length; the
            // backing slice contains `size.div_ceil(W::BITS)` words,
            // which is exactly what `std::slice::ChunksMut` hands us.
            let next = unsafe { BitVec::from_raw_parts(chunk, size) };
            self.remaining -= size;
            next
        })
    }
}

impl<'a, W: Word> ExactSizeIterator for BitVecChunksMut<'a, W> where
    std::slice::ChunksMut<'a, W>: ExactSizeIterator
{
}

impl<'a, W: Word> FusedIterator for BitVecChunksMut<'a, W> where
    std::slice::ChunksMut<'a, W>: FusedIterator
{
}

/// Error returned when [`BitVec::try_chunks_mut`] cannot align the
/// requested chunk size to word boundaries.
///
/// [`BitVec::try_chunks_mut`]: SliceByValueMut::try_chunks_mut
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BitVecChunksMutError<W: Word> {
    chunk_size: usize,
    _marker: core::marker::PhantomData<W>,
}

impl<W: Word> fmt::Display for BitVecChunksMutError<W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "try_chunks_mut needs the chunk size ({}) to be a multiple of W::BITS ({}) to return more than one chunk",
            self.chunk_size,
            W::BITS as usize
        )
    }
}

impl<W: Word> std::error::Error for BitVecChunksMutError<W> {}

impl<B: Backend<Word: Word> + AsRef<[B::Word]>> SliceByValue for BitVec<B> {
    type Value = B::Word;

    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    unsafe fn get_value_unchecked(&self, index: usize) -> B::Word {
        // Delegate to the canonical `BitVecOps::get_unchecked`; LLVM
        // collapses the `if` to the same `(word >> bit) & 1` it would
        // emit for a direct implementation.
        if unsafe { self.get_unchecked(index) } {
            B::Word::ONE
        } else {
            B::Word::ZERO
        }
    }
}

impl<B: Backend<Word: Word>> BitWidth for BitVec<B> {
    #[inline(always)]
    fn bit_width(&self) -> usize {
        1
    }
}

impl<B: Backend<Word: Word> + AsRef<[B::Word]>> BitFieldSlice for BitVec<B> {
    #[inline(always)]
    fn as_slice(&self) -> &[Self::Value] {
        self.bits.as_ref()
    }
}

impl<B: Backend<Word: Word> + AsRef<[B::Word]> + AsMut<[B::Word]>> SliceByValueMut for BitVec<B> {
    #[inline(always)]
    unsafe fn set_value_unchecked(&mut self, index: usize, value: B::Word) {
        // Delegate to `BitVecOpsMut::set_unchecked`: its `if value`
        // form dead-code-eliminates to a single RMW when the caller
        // passes a compile-time constant, and ties my hand-rolled
        // branchless form on random-runtime values.
        unsafe { self.set_unchecked(index, value != B::Word::ZERO) }
    }

    type ChunksMut<'a>
        = BitVecChunksMut<'a, B::Word>
    where
        Self: 'a;

    type ChunksMutError = BitVecChunksMutError<B::Word>;

    /// # Errors
    ///
    /// Returns an error if `chunk_size` is not a multiple of
    /// `W::BITS` and more than one chunk must be returned.
    fn try_chunks_mut(
        &mut self,
        chunk_size: usize,
    ) -> Result<Self::ChunksMut<'_>, BitVecChunksMutError<B::Word>> {
        let len = self.len;
        let bits = B::Word::BITS as usize;
        if len <= chunk_size || chunk_size % bits == 0 {
            // `std::slice::ChunksMut::new` panics on chunk_size 0, so
            // use 1 when the chunk is empty; the iterator will yield
            // empty views anyway.
            let words_per_chunk = Ord::max(1, chunk_size.div_ceil(bits));
            Ok(BitVecChunksMut {
                remaining: len,
                chunk_size,
                iter: self.bits.as_mut()[..len.div_ceil(bits)].chunks_mut(words_per_chunk),
            })
        } else {
            Err(BitVecChunksMutError {
                chunk_size,
                _marker: core::marker::PhantomData,
            })
        }
    }
}

impl<B: Backend<Word: Word> + AsRef<[B::Word]> + AsMut<[B::Word]>> BitFieldSliceMut for BitVec<B> {
    fn reset(&mut self) {
        <Self as BitVecOpsMut<B::Word>>::fill(self, false);
    }

    #[cfg(feature = "rayon")]
    fn par_reset(&mut self) {
        use rayon::prelude::*;
        let bits_per_word = B::Word::BITS as usize;
        let full_words = self.len / bits_per_word;
        let residual = self.len % bits_per_word;
        let data = self.bits.as_mut();
        data[..full_words]
            .par_iter_mut()
            .with_min_len(crate::RAYON_MIN_LEN)
            .for_each(|x| *x = B::Word::ZERO);
        if residual != 0 {
            data[full_words] &= B::Word::MAX << residual;
        }
    }

    #[inline(always)]
    fn as_mut_slice(&mut self) -> &mut [Self::Value] {
        self.bits.as_mut()
    }
}

#[derive(Debug, Clone, MemSize, MemDbg, Delegate)]
/// A thread-safe bit vector.
///
/// See the [module documentation] for details.
///
/// [module documentation]: mod@crate::bits::bit_vec
#[delegate(crate::traits::Backend, target = "bits")]
pub struct AtomicBitVec<B = Box<[Atomic<usize>]>> {
    bits: B,
    len: usize,
}

impl<B> AtomicBitVec<B> {
    /// Returns the number of bits in the bit vector.
    ///
    /// This method is equivalent to [`BitLength::len`], but it is provided to
    /// reduce ambiguity in method resolution.
    #[inline(always)]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// # Safety
    /// `len` must be between 0 (included) and the number of
    /// bits in `bits` (included).
    #[inline(always)]
    #[must_use]
    pub const unsafe fn from_raw_parts(bits: B, len: usize) -> Self {
        Self { bits, len }
    }
    /// Returns the backend and the length in bits, consuming this bit vector.
    #[inline(always)]
    pub fn into_raw_parts(self) -> (B, usize) {
        (self.bits, self.len)
    }
}

impl<B: Backend<Word: PrimitiveAtomicUnsigned<Value: Word>> + From<Vec<B::Word>>> AtomicBitVec<B> {
    /// Creates a new atomic bit vector of length `len` initialized to `false`.
    #[must_use]
    pub fn new(len: usize) -> Self {
        Self::with_value(len, false)
    }

    /// Creates a new atomic bit vector of length `len` initialized to `value`.
    #[must_use]
    pub fn with_value(len: usize, value: bool) -> Self {
        let bits_per_word = <B::Word as PrimitiveAtomic>::Value::BITS as usize;
        let n_of_words = len.div_ceil(bits_per_word);
        let extra_bits = (n_of_words * bits_per_word) - len;
        let word_value = if value {
            !<B::Word as PrimitiveAtomic>::Value::ZERO
        } else {
            <B::Word as PrimitiveAtomic>::Value::ZERO
        };
        let mut bits: Vec<B::Word> = (0..n_of_words).map(|_| B::Word::new(word_value)).collect();
        if extra_bits > 0 {
            let last_word_value = word_value >> extra_bits;
            bits[n_of_words - 1] = B::Word::new(last_word_value);
        }
        Self {
            bits: B::from(bits),
            len,
        }
    }
}

impl<B> BitLength for AtomicBitVec<B> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }
}

impl<B: Backend<Word: PrimitiveAtomicUnsigned<Value: Word>> + AsRef<[B::Word]>> BitCount
    for AtomicBitVec<B>
{
    fn count_ones(&self) -> usize {
        let bits_per_word = <B::Word as PrimitiveAtomic>::Value::BITS as usize;
        let full_words = self.len() / bits_per_word;
        let residual = self.len() % bits_per_word;
        let bits: &[B::Word] = self.as_ref();
        let mut num_ones;
        // Just to be sure, add a fence to ensure that we will see all the final
        // values
        core::sync::atomic::fence(Ordering::SeqCst);
        num_ones = bits[..full_words]
            .iter()
            .map(|x| x.load(Ordering::Relaxed).count_ones() as usize)
            .sum();
        if residual != 0 {
            num_ones += (bits[full_words].load(Ordering::Relaxed) << (bits_per_word - residual))
                .count_ones() as usize
        }
        num_ones
    }
}

impl<B: Backend<Word: PrimitiveAtomicUnsigned<Value: Word>> + AsRef<[B::Word]>> Index<usize>
    for AtomicBitVec<B>
{
    type Output = bool;

    /// Shorthand for `get` using [`Ordering::Relaxed`].
    fn index(&self, index: usize) -> &Self::Output {
        match AtomicBitVecOps::<B::Word>::get(self, index, Ordering::Relaxed) {
            false => &false,
            true => &true,
        }
    }
}

impl<'a, B: Backend<Word: PrimitiveAtomicUnsigned<Value: Word>> + AsRef<[B::Word]>> IntoIterator
    for &'a AtomicBitVec<B>
{
    type IntoIter = AtomicBitIter<'a, B::Word, B>;
    type Item = bool;

    fn into_iter(self) -> Self::IntoIter {
        AtomicBitIter::new(&self.bits, self.len())
    }
}

// Conversions

/// This conversion may fail if the alignment of `W` is not the same as
/// that of `W::Atomic`.
impl<'a, W: AtomicPrimitive> TryFrom<BitVec<&'a [W]>> for AtomicBitVec<&'a [W::Atomic]> {
    type Error = CannotCastToAtomicError<W>;
    fn try_from(value: BitVec<&'a [W]>) -> Result<Self, Self::Error> {
        if core::mem::align_of::<W>() != core::mem::align_of::<W::Atomic>() {
            return Err(CannotCastToAtomicError::default());
        }
        Ok(AtomicBitVec {
            bits: unsafe { core::mem::transmute::<&'a [W], &'a [W::Atomic]>(value.bits) },
            len: value.len,
        })
    }
}

/// This conversion may fail if the alignment of `W` is not the same as
/// that of `W::Atomic`.
impl<'a, W: AtomicPrimitive> TryFrom<BitVec<&'a mut [W]>> for AtomicBitVec<&'a mut [W::Atomic]> {
    type Error = CannotCastToAtomicError<W>;
    fn try_from(value: BitVec<&'a mut [W]>) -> Result<Self, Self::Error> {
        if core::mem::align_of::<W>() != core::mem::align_of::<W::Atomic>() {
            return Err(CannotCastToAtomicError::default());
        }
        Ok(AtomicBitVec {
            bits: unsafe { core::mem::transmute::<&'a mut [W], &'a mut [W::Atomic]>(value.bits) },
            len: value.len,
        })
    }
}

impl<W: AtomicPrimitive> From<AtomicBitVec<Box<[W::Atomic]>>> for BitVec<Vec<W>> {
    fn from(value: AtomicBitVec<Box<[W::Atomic]>>) -> Self {
        BitVec {
            bits: transmute_vec_from_atomic::<W::Atomic>(value.bits.into_vec()),
            len: value.len,
        }
    }
}

impl<W: AtomicPrimitive> From<BitVec<Vec<W>>> for AtomicBitVec<Box<[W::Atomic]>> {
    fn from(value: BitVec<Vec<W>>) -> Self {
        AtomicBitVec {
            bits: transmute_vec_into_atomic(value.bits).into_boxed_slice(),
            len: value.len,
        }
    }
}

impl<W: AtomicPrimitive> From<AtomicBitVec<Box<[W::Atomic]>>> for BitVec<Box<[W]>> {
    fn from(value: AtomicBitVec<Box<[W::Atomic]>>) -> Self {
        BitVec {
            bits: transmute_boxed_slice_from_atomic::<W::Atomic>(value.bits),
            len: value.len,
        }
    }
}

impl<W: AtomicPrimitive + Copy> From<BitVec<Box<[W]>>> for AtomicBitVec<Box<[W::Atomic]>> {
    fn from(value: BitVec<Box<[W]>>) -> Self {
        AtomicBitVec {
            bits: transmute_boxed_slice_into_atomic::<W>(value.bits),
            len: value.len,
        }
    }
}

impl<'a, W: AtomicPrimitive> From<AtomicBitVec<&'a [W::Atomic]>> for BitVec<&'a [W]> {
    fn from(value: AtomicBitVec<&'a [W::Atomic]>) -> Self {
        BitVec {
            bits: unsafe { core::mem::transmute::<&'a [W::Atomic], &'a [W]>(value.bits) },
            len: value.len,
        }
    }
}

impl<'a, W: AtomicPrimitive> From<AtomicBitVec<&'a mut [W::Atomic]>> for BitVec<&'a mut [W]> {
    fn from(value: AtomicBitVec<&'a mut [W::Atomic]>) -> Self {
        BitVec {
            bits: unsafe { core::mem::transmute::<&'a mut [W::Atomic], &'a mut [W]>(value.bits) },
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

impl<W: Word, B: AsRef<[W]>> AsRef<[W]> for BitVec<B> {
    #[inline(always)]
    fn as_ref(&self) -> &[W] {
        self.bits.as_ref()
    }
}

impl<W: Word, B: AsMut<[W]>> AsMut<[W]> for BitVec<B> {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut [W] {
        self.bits.as_mut()
    }
}

impl<B: Backend + AsRef<[B::Word]>> AsRef<[B::Word]> for AtomicBitVec<B> {
    #[inline(always)]
    fn as_ref(&self) -> &[B::Word] {
        self.bits.as_ref()
    }
}

impl<B: Backend<Word: Word> + AsRef<[B::Word]>> RankHinted for BitVec<B> {
    #[inline(always)]
    unsafe fn rank_hinted<const WORDS_PER_SUBBLOCK: usize>(
        &self,
        pos: usize,
        hint_pos: usize,
        hint_rank: usize,
    ) -> usize {
        let bits_per_word = B::Word::BITS as usize;
        let bits: &[B::Word] = self.as_ref();
        let mut rank = hint_rank;
        let mut hp = hint_pos;

        debug_assert!(hp < bits.len(), "hint_pos: {}, len: {}", hp, bits.len());

        // Prefetch the word containing `pos` so that the load below overlaps
        // with the popcount accumulation loop — but only when the subblock
        // spans more than one cache line.  When the subblock fits in a single
        // 64-byte cache line (e.g. 8 × u64 words), the first loop iteration
        // already pulls in the target word and the prefetch is pure overhead.
        if WORDS_PER_SUBBLOCK * std::mem::size_of::<B::Word>() > 64 {
            crate::utils::prefetch_index(bits, pos / bits_per_word);
        }

        if WORDS_PER_SUBBLOCK == usize::MAX {
            // Unbounded: fall back to while loop (used when the caller cannot
            // provide a compile-time bound).
            while (hp + 1) * bits_per_word <= pos {
                rank += unsafe { bits.get_unchecked(hp) }.count_ones() as usize;
                hp += 1;
            }
        } else {
            // Bounded: the loop runs at most WORDS_PER_SUBBLOCK-1 times.
            // LLVM can fully unroll this when WORDS_PER_SUBBLOCK is small.
            for _ in 0..WORDS_PER_SUBBLOCK - 1 {
                if (hp + 1) * bits_per_word > pos {
                    break;
                }
                rank += unsafe { bits.get_unchecked(hp) }.count_ones() as usize;
                hp += 1;
            }
        }

        rank + (unsafe { *bits.get_unchecked(hp) }
            & (B::Word::ONE << (pos % bits_per_word) as u32).wrapping_sub(B::Word::ONE))
        .count_ones() as usize
    }
}

// SelectHinted and SelectZeroHinted for BitVec.

impl<B: Backend<Word: Word + SelectInWord> + AsRef<[B::Word]>> SelectHinted for BitVec<B> {
    #[inline(always)]
    unsafe fn select_hinted<const WORDS_PER_SUBBLOCK: usize>(
        &self,
        rank: usize,
        hint_pos: usize,
        hint_rank: usize,
    ) -> usize {
        let bits_per_word = B::Word::BITS as usize;
        let bits: &[B::Word] = self.as_ref();
        let mut word_index = hint_pos / bits_per_word;
        let bit_index = hint_pos % bits_per_word;
        let mut residual = rank - hint_rank;
        let mut word = (unsafe { *bits.get_unchecked(word_index) } >> bit_index) << bit_index;
        // WORDS_PER_SUBBLOCK == usize::MAX means unbounded (caller doesn't know the bound).
        // Otherwise the loop runs at most WORDS_PER_SUBBLOCK times, helping LLVM unroll.
        let limit = if WORDS_PER_SUBBLOCK == usize::MAX {
            usize::MAX
        } else {
            WORDS_PER_SUBBLOCK
        };
        for _ in 0..limit {
            let bit_count = word.count_ones() as usize;
            if residual < bit_count {
                return word_index * bits_per_word + word.select_in_word(residual);
            }
            word_index += 1;
            word = *unsafe { bits.get_unchecked(word_index) };
            residual -= bit_count;
        }
        unreachable!()
    }
}

impl<B: Backend<Word: Word + SelectInWord> + AsRef<[B::Word]>> SelectZeroHinted for BitVec<B> {
    #[inline(always)]
    unsafe fn select_zero_hinted<const WORDS_PER_SUBBLOCK: usize>(
        &self,
        rank: usize,
        hint_pos: usize,
        hint_rank: usize,
    ) -> usize {
        let bits_per_word = B::Word::BITS as usize;
        let bits: &[B::Word] = self.as_ref();
        let mut word_index = hint_pos / bits_per_word;
        let bit_index = hint_pos % bits_per_word;
        let mut residual = rank - hint_rank;
        let mut word = (!unsafe { *bits.get_unchecked(word_index) } >> bit_index) << bit_index;
        let limit = if WORDS_PER_SUBBLOCK == usize::MAX {
            usize::MAX
        } else {
            WORDS_PER_SUBBLOCK
        };
        for _ in 0..limit {
            let bit_count = word.count_ones() as usize;
            if residual < bit_count {
                return word_index * bits_per_word + word.select_in_word(residual);
            }
            word_index += 1;
            word = unsafe { !*bits.get_unchecked(word_index) };
            residual -= bit_count;
        }
        unreachable!()
    }
}

/// A wrapper around [`BitVec`] that implements [`BitVecValueOps`] using
/// unaligned reads.
///
/// Obtain an instance via [`TryIntoUnaligned`] on a `BitVec<Box<[W]>>`,
/// which adds a padding word if one is not already present. You can recover
/// the original [`BitVec`] using a [`From` implementation]
///
/// [`From` implementation]: #impl-From<BitVecU<Box<%5BW%5D>>-for-BitVec<Box<%5BW%5D>>
/// [`TryIntoUnaligned`]: crate::traits::TryIntoUnaligned
///
/// Note that unaligned reads give correct results only when the bit width
/// satisfies the unaligned constraints (at most `W::BITS - 6`, or exactly
/// `W::BITS - 4`, or exactly `W::BITS`). Using other widths will not
/// cause undefined behavior, but may return incorrect values.
///
/// We delegate [`Backend`], [`BitLength`], and
/// [`AsRef<[Backend::Word]>`](core::convert::AsRef) to make [`BitVecOps`]
/// methods available, and [`Index`] to make slice-like read-only access
/// available.
#[derive(Debug, Clone, MemSize, MemDbg, Delegate)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[delegate(Index<usize>, target = "0")]
#[delegate(crate::traits::Backend, target = "0")]
#[delegate(crate::traits::bit_vec_ops::BitLength, target = "0")]
pub struct BitVecU<B>(BitVec<B>);

impl<W: Word> From<BitVecU<Box<[W]>>> for BitVec<Box<[W]>> {
    /// Converts a [`BitVecU`] back into a [`BitVec`].
    ///
    /// The padding word is kept in the backing storage so that a subsequent
    /// [`try_into_unaligned`] does not need to reallocate.
    ///
    /// [`try_into_unaligned`]: crate::traits::TryIntoUnaligned::try_into_unaligned
    fn from(unaligned: BitVecU<Box<[W]>>) -> Self {
        unaligned.0
    }
}

impl<W: Word> crate::traits::TryIntoUnaligned for BitVec<Box<[W]>> {
    type Unaligned = BitVecU<Box<[W]>>;

    fn try_into_unaligned(
        self,
    ) -> Result<Self::Unaligned, crate::traits::UnalignedConversionError> {
        let needed = self.len().div_ceil(W::BITS as usize);
        if self.as_ref().len() > needed {
            Ok(BitVecU(self))
        } else {
            let (raw, len) = self.into_raw_parts();
            let mut v = raw.into_vec();
            v.reserve_exact(1);
            v.push(W::ZERO);
            Ok(BitVecU(unsafe {
                BitVec::from_raw_parts(v.into_boxed_slice(), len)
            }))
        }
    }
}

impl<B: Backend<Word: Word> + AsRef<[B::Word]>> BitVecValueOps<B::Word> for BitVecU<B> {
    #[inline(always)]
    fn get_value(&self, pos: usize, width: usize) -> B::Word {
        self.0.get_value_unaligned(pos, width)
    }

    #[inline(always)]
    unsafe fn get_value_unchecked(&self, pos: usize, width: usize) -> B::Word {
        unsafe { self.0.get_value_unaligned_unchecked(pos, width) }
    }
}

impl<B: Backend + AsRef<[B::Word]>> AsRef<[B::Word]> for BitVecU<B> {
    #[inline(always)]
    fn as_ref(&self) -> &[B::Word] {
        self.0.bits.as_ref()
    }
}
