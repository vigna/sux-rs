/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Vectors of values of fixed bit width.
//!
//! Elements are stored contiguously, with no padding bits (in particular,
//! unless the bit width is a power of two some elements will be stored across
//! word boundaries).
//!
//! There are two flavors: [`BitFieldVec`], a mutable bit-field vector, and
//! [`AtomicBitFieldVec`], a mutable, thread-safe bit-field vector.
//!
//! These flavors depends on a backend, and presently we provide, given an
//! unsigned integer type `W` or an unsigned atomic integer type `A`:
//!
//! - `BitFieldVec<Vec<T>>`: a mutable, growable and resizable bit-field vector;
//! - `BitFieldVec<AsRef<[W]>>`: an immutable bit-field vector, useful for
//!   [ε-serde](epserde) support;
//! - `BitFieldVec<AsRef<[W]> + AsMut<[W]>>`: a mutable (but not resizable) bit
//!    vector;
//! - `AtomicBitFieldVec<AsRef<[A]>>`: a partially thread-safe, mutable (but not
//!   resizable) bit-field vector.
//!
//! More generally, the underlying type must satisfy the trait [`Word`] for
//! [`BitFieldVec`] and additionally [`IntoAtomic`] for [`AtomicBitFieldVec`].
//! A blanket implementation exposes slices of elements of type `W` as bit-field
//! vectors of width `W::BITS`, analogously for atomic types `A`.
//!
//! Note that nothing is assumed about the content of the backend outside the
//! bits of the vector. Moreover, the content of the backend outside of the
//! vector is never modified by the methods of this structure.
//!
//! For high-speed unchecked scanning, we implement [`IntoUncheckedIterator`]
//! and [`IntoReverseUncheckedIterator`] on a reference to this type. The are
//! used, for example, to provide
//! [predecessor](crate::traits::indexed_dict::Pred) and
//! [successor](crate::traits::indexed_dict::Succ) primitives for
//! [`EliasFano`].
//!
//! # Low-level support
//!
//! The methods [`address_of`](BitFieldVec::addr_of) and
//! [`get_unaligned`](BitFieldVec::get_unaligned) can be used to manually
//! prefetch parts of the data structure, or read values using unaligned read,
//! when the bit width makes it possible.
//!
//! # Examples
//! ```rust
//! use sux::prelude::*;
//!
//! // Bit field vector of bit width 5 and length 10, all entries set to zero
//! let mut b = <BitFieldVec<usize>>::new(5, 10);
//! assert_eq!(b.len(), 10);
//! assert_eq!(b.bit_width(), 5);
//! b.set(0, 3);
//! assert_eq!(b.get(0), 3);
//!
//! // Empty bit field vector of bit width 20 with capacity 10
//! let mut b = <BitFieldVec<usize>>::with_capacity(20, 10);
//! assert_eq!(b.len(), 0);
//! assert_eq!(b.bit_width(), 20);
//! b.push(20);
//! assert_eq!(b.len(), 1);
//! assert_eq!(b.get(0), 20);
//! assert_eq!(b.pop(), Some(20));
//!
//! // Convenience macro
//! let b = bit_field_vec![10; 4, 500, 2, 0, 1];
//! assert_eq!(b.len(), 5);
//! assert_eq!(b.bit_width(), 10);
//! assert_eq!(b.get(0), 4);
//! assert_eq!(b.get(1), 500);
//! assert_eq!(b.get(2), 2);
//! assert_eq!(b.get(3), 0);
//! assert_eq!(b.get(4), 1);
//! ```

use crate::prelude::*;
use crate::traits::bit_field_slice::{panic_if_out_of_bounds, panic_if_value};
use crate::utils::{transmute_boxed_slice, transmute_vec};
use anyhow::{bail, Result};
use common_traits::*;
use epserde::*;
use mem_dbg::*;
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use std::sync::atomic::*;

/// Convenient, [`vec!`]-like macro to initialize `usize`-based bit-field
/// vectors.
///
/// Note that the syntax `bit_field_vec![width; length; value]` that has been
/// deprecated in favor of `bit_field_vec![width => value; length]`, so that
/// value and length are in the same order as in [`vec!`].
///
/// # Examples
///
/// ```rust
/// use sux::prelude::*;
///
/// // Empty bit field vector of bit width 5
/// let b = bit_field_vec![5];
/// assert_eq!(b.len(), 0);
/// assert_eq!(b.bit_width(), 5);
///
/// // 10 values of bit width 6, all set to 3
/// let b = bit_field_vec![6 => 3; 10];
/// assert_eq!(b.len(), 10);
/// assert_eq!(b.bit_width(), 6);
/// assert_eq!(b.iter().all(|x| x == 3), true);
///
/// // List of values of bit width 10
/// let b = bit_field_vec![10; 4, 500, 2, 0, 1];
/// assert_eq!(b.len(), 5);
/// assert_eq!(b.bit_width(), 10);
/// assert_eq!(b.get(0), 4);
/// assert_eq!(b.get(1), 500);
/// assert_eq!(b.get(2), 2);
/// assert_eq!(b.get(3), 0);
/// assert_eq!(b.get(4), 1);
/// ```
#[macro_export]
macro_rules! bit_field_vec {
    ($w:expr) => {
        $crate::bits::BitFieldVec::<usize, _>::new($w, 0)
    };
    ($w:expr; $n:expr; $v:expr) => {
        {
            let mut bit_field_vec = $crate::bits::BitFieldVec::<usize, _>::with_capacity($w, $n);
            // Force type
            let v: usize = $v;
            bit_field_vec.resize($n, v);
            bit_field_vec
        }
    };
    ($w:expr => $v:expr; $n:expr) => {
        {
            let mut bit_field_vec = $crate::bits::BitFieldVec::<usize, _>::with_capacity($w, $n);
            // Force type
            let v: usize = $v;
            bit_field_vec.resize($n, v);
            bit_field_vec
        }
    };
    ($w:expr; $($x:expr),+ $(,)?) => {
        {
            let mut b = $crate::bits::BitFieldVec::<usize, _>::with_capacity($w, [$($x),+].len());
            $(
                // Force type
                let x: usize = $x;
                b.push(x);
            )*
            b
        }
    };
}

/// A vector of bit fields of fixed width.
#[derive(Epserde, Debug, Clone, Hash, MemDbg, MemSize)]
pub struct BitFieldVec<W: Word = usize, B = Vec<W>> {
    /// The underlying storage.
    bits: B,
    /// The bit width of the values stored in the vector.
    bit_width: usize,
    /// A mask with its lowest `bit_width` bits set to one.
    mask: W,
    /// The length of the vector.
    len: usize,
}

fn mask<W: Word>(bit_width: usize) -> W {
    if bit_width == 0 {
        W::ZERO
    } else {
        W::MAX >> (W::BITS - bit_width)
    }
}

impl<W: Word, B> BitFieldVec<W, B> {
    /// # Safety
    /// `len` * `bit_width` must be between 0 (included) the number of
    /// bits in `bits` (included).
    #[inline(always)]
    pub unsafe fn from_raw_parts(bits: B, bit_width: usize, len: usize) -> Self {
        Self {
            bits,
            bit_width,
            mask: mask(bit_width),
            len,
        }
    }

    #[inline(always)]
    pub fn into_raw_parts(self) -> (B, usize, usize) {
        (self.bits, self.bit_width, self.len)
    }

    #[inline(always)]
    /// Modifies the bit field in place.
    ///
    /// # Safety
    /// This is unsafe because it's the caller's responsibility to ensure that
    /// that the length is compatible with the modified bits.
    pub unsafe fn map<W2: Word, B2>(self, f: impl FnOnce(B) -> B2) -> BitFieldVec<W2, B2> {
        BitFieldVec {
            bits: f(self.bits),
            bit_width: self.bit_width,
            mask: mask(self.bit_width),
            len: self.len,
        }
    }
}

impl<W: Word, B: AsRef<[W]>> BitFieldVec<W, B> {
    /// Gets the address of the item storing (the first part of)
    /// the element of given index.
    ///
    /// This method is mainly useful for manually prefetching
    /// parts of the data structure.
    pub fn addr_of(&self, index: usize) -> *const W {
        let start_bit = index * self.bit_width;
        let word_index = start_bit / W::BITS;
        (&self.bits.as_ref()[word_index]) as *const _
    }

    /// Like [`BitFieldSlice::get`], but using unaligned reads.
    ///
    /// This method can be used for bit width smaller than or equal to `W::BITS
    /// - 8 + 2` or equal to `W::BITS - 8 + 4` or `W::BITS`.
    pub fn get_unaligned(&self, index: usize) -> W {
        panic_if_out_of_bounds!(index, self.len);
        unsafe { self.get_unaligned_unchecked(index) }
    }

    /// Like [`BitFieldSlice::get`], but using unaligned reads.
    ///
    /// # Safety
    ///
    /// This method can be used for bit width smaller than or equal
    /// to `W::BITS - 8 + 2` or equal to `W::BITS - 8 + 4` or `W::BITS`.
    pub unsafe fn get_unaligned_unchecked(&self, index: usize) -> W {
        debug_assert!(
            self.bit_width <= W::BITS - 8 + 2
                || self.bit_width == W::BITS - 8 + 4
                || self.bit_width == W::BITS
        );
        let base_ptr = self.bits.as_ref().as_ptr() as *const u8;
        let start_bit = index * self.bit_width;
        let ptr = base_ptr.add(start_bit / W::BYTES) as *const W;
        let word = core::ptr::read_unaligned(ptr);
        (word >> (start_bit % 8)) & self.mask
    }

    /// Returns the backend of the vector as a slice of `W`.
    pub fn as_slice(&self) -> &[W] {
        self.bits.as_ref()
    }
}

impl<W: Word, B: AsMut<[W]>> BitFieldVec<W, B> {
    /// Returns the backend of the vector as a mutable slice of `W`.
    pub fn as_mut_slice(&mut self) -> &mut [W] {
        self.bits.as_mut()
    }
}

impl<W: Word, B: AsRef<[W]>> BitFieldVec<W, B> {}

impl<W: Word> BitFieldVec<W, Vec<W>> {
    /// Creates a new zero-initialized vector of given bit width and length.
    pub fn new(bit_width: usize, len: usize) -> Self {
        // We need at least one word to handle the case of bit width zero.
        let n_of_words = Ord::max(1, (len * bit_width).div_ceil(W::BITS));
        Self {
            bits: vec![W::ZERO; n_of_words],
            bit_width,
            mask: mask(bit_width),
            len,
        }
    }

    /// Creates an empty vector that doesn't need to reallocate for up to
    /// `capacity` elements.
    pub fn with_capacity(bit_width: usize, capacity: usize) -> Self {
        // We need at least one word to handle the case of bit width zero.
        let n_of_words = Ord::max(1, (capacity * bit_width).div_ceil(W::BITS));
        Self {
            bits: Vec::with_capacity(n_of_words),
            bit_width,
            mask: mask(bit_width),
            len: 0,
        }
    }

    /// Sets the length.
    ///
    /// # Safety
    ///
    /// `len * bit_width` must be at most `self.bits.len() * W::BITS`. Note that
    /// setting the length might result in reading uninitialized data.
    pub unsafe fn set_len(&mut self, len: usize) {
        debug_assert!(len * self.bit_width <= self.bits.len() * W::BITS);
        self.len = len;
    }

    /// Sets len to 0
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Returns the bit width of the values inside the vector.
    pub fn bit_width(&self) -> usize {
        debug_assert!(self.bit_width <= W::BITS);
        self.bit_width
    }

    /// Returns the mask used to extract values from the vector.
    /// This will keep the lowest `bit_width` bits.
    pub fn mask(&self) -> W {
        self.mask
    }

    /// Creates a new vector by copying a slice; the bit width will be the minimum
    /// width sufficient to hold all values in the slice.
    ///
    /// Returns an error if the bit width of the values in `slice` is larger than
    /// `W::BITS`.
    pub fn from_slice<SW>(slice: &impl BitFieldSlice<SW>) -> Result<Self>
    where
        SW: Word + CastableInto<W>,
    {
        let mut max_len: usize = 0;
        for i in 0..slice.len() {
            max_len = Ord::max(max_len, unsafe { slice.get_unchecked(i).len() as usize });
        }

        if max_len > W::BITS {
            bail!(
                "Cannot convert a slice of bit width {} into a slice with W = {}",
                max_len,
                std::any::type_name::<W>()
            );
        }
        let mut result = Self::new(max_len, slice.len());
        for i in 0..slice.len() {
            unsafe { result.set_unchecked(i, slice.get_unchecked(i).cast()) };
        }

        Ok(result)
    }

    /// Adds a value at the end of the vector.
    pub fn push(&mut self, value: W) {
        panic_if_value!(value, self.mask, self.bit_width);
        if (self.len + 1) * self.bit_width > self.bits.len() * W::BITS {
            self.bits.push(W::ZERO);
        }
        unsafe {
            self.set_unchecked(self.len, value);
        }
        self.len += 1;
    }

    /// Truncates or exted with `value` the vector.
    pub fn resize(&mut self, new_len: usize, value: W) {
        panic_if_value!(value, self.mask, self.bit_width);
        if new_len > self.len {
            if new_len * self.bit_width > self.bits.len() * W::BITS {
                self.bits
                    .resize((new_len * self.bit_width).div_ceil(W::BITS), W::ZERO);
            }
            for i in self.len..new_len {
                unsafe {
                    self.set_unchecked(i, value);
                }
            }
        }
        self.len = new_len;
    }

    /// Removes and returns a value from the end of the vector.
    ///
    /// Returns None if the [`BitFieldVec`] is empty.
    pub fn pop(&mut self) -> Option<W> {
        if self.len == 0 {
            return None;
        }
        let value = self.get(self.len - 1);
        self.len -= 1;
        Some(value)
    }
}

impl<W: Word, T> BitFieldSliceCore<W> for BitFieldVec<W, T> {
    #[inline(always)]
    fn bit_width(&self) -> usize {
        debug_assert!(self.bit_width <= W::BITS);
        self.bit_width
    }
    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }
}

impl<W: Word, B: AsRef<[W]>> BitFieldSlice<W> for BitFieldVec<W, B> {
    #[inline]
    unsafe fn get_unchecked(&self, index: usize) -> W {
        let pos = index * self.bit_width;
        let word_index = pos / W::BITS;
        let bit_index = pos % W::BITS;
        let bits = self.bits.as_ref();

        if bit_index + self.bit_width <= W::BITS {
            (*bits.get_unchecked(word_index) >> bit_index) & self.mask
        } else {
            ((*bits.get_unchecked(word_index) >> bit_index)
                | (*bits.get_unchecked(word_index + 1) << (W::BITS - bit_index)))
                & self.mask
        }
    }
}

impl<W: Word, B: AsRef<[W]> + AsMut<[W]>> BitFieldSliceMut<W> for BitFieldVec<W, B> {
    #[inline(always)]
    fn mask(&self) -> W {
        self.mask
    }

    #[inline(always)]
    fn set(&mut self, index: usize, value: W) {
        panic_if_out_of_bounds!(index, self.len);
        panic_if_value!(value, self.mask, self.bit_width);
        unsafe {
            self.set_unchecked(index, value);
        }
    }

    #[inline]
    unsafe fn set_unchecked(&mut self, index: usize, value: W) {
        let pos = index * self.bit_width;
        let word_index = pos / W::BITS;
        let bit_index = pos % W::BITS;
        let bits = self.bits.as_mut();

        if bit_index + self.bit_width <= W::BITS {
            let mut word = *bits.get_unchecked_mut(word_index);
            word &= !(self.mask << bit_index);
            word |= value << bit_index;
            *bits.get_unchecked_mut(word_index) = word;
        } else {
            let mut word = *bits.get_unchecked_mut(word_index);
            word &= (W::ONE << bit_index) - W::ONE;
            word |= value << bit_index;
            *bits.get_unchecked_mut(word_index) = word;

            let mut word = *bits.get_unchecked_mut(word_index + 1);
            word &= !(self.mask >> (W::BITS - bit_index));
            word |= value >> (W::BITS - bit_index);
            *bits.get_unchecked_mut(word_index + 1) = word;
        }
    }

    fn reset(&mut self) {
        let bit_len = self.len * self.bit_width;
        let full_words = bit_len / W::BITS;
        let residual = bit_len % W::BITS;
        let bits = self.bits.as_mut();

        #[cfg(feature = "rayon")]
        {
            bits[..full_words].par_iter_mut().for_each(|x| *x = W::ZERO);
        }

        #[cfg(not(feature = "rayon"))]
        {
            bits[..full_words].iter_mut().for_each(|x| *x = W::ZERO);
        }

        if residual != 0 {
            bits[full_words] &= W::MAX << residual;
        }
    }

    /// This implementation perform the copy word by word, which is
    /// significantly faster than the default implementation.
    fn copy(&self, from: usize, dst: &mut Self, to: usize, len: usize) {
        assert_eq!(
            self.bit_width, dst.bit_width,
            "Bit widths must be equal (self: {}, dest: {})",
            self.bit_width, dst.bit_width
        );
        // Reduce len to the elements available in both vectors
        let len = Ord::min(Ord::min(len, dst.len - to), self.len - from);
        if len == 0 {
            return;
        }
        let bit_width = Ord::min(self.bit_width, dst.bit_width);
        let bit_len = len * bit_width;
        let src_pos = from * self.bit_width;
        let dst_pos = to * dst.bit_width;
        let src_bit = src_pos % W::BITS;
        let dst_bit = dst_pos % W::BITS;
        let src_first_word = src_pos / W::BITS;
        let dst_first_word = dst_pos / W::BITS;
        let src_last_word = (src_pos + bit_len - 1) / W::BITS;
        let dst_last_word = (dst_pos + bit_len - 1) / W::BITS;
        let source = self.bits.as_ref();
        let dest = dst.bits.as_mut();

        if src_first_word == src_last_word && dst_first_word == dst_last_word {
            let mask = W::MAX >> (W::BITS - bit_len);
            let word = source[src_first_word] >> src_bit & mask;
            dest[dst_first_word] &= !(mask << dst_bit);
            dest[dst_first_word] |= word << dst_bit;
        } else if src_first_word == src_last_word {
            // dst_first_word != dst_last_word
            let mask = W::MAX >> (W::BITS - bit_len);
            let word = source[src_first_word] >> src_bit & mask;
            dest[dst_first_word] &= !(mask << dst_bit);
            dest[dst_first_word] |= (word & mask) << dst_bit;
            dest[dst_last_word] &= !(mask >> (W::BITS - dst_bit));
            dest[dst_last_word] |= (word & mask) >> (W::BITS - dst_bit);
        } else if dst_first_word == dst_last_word {
            // src_first_word != src_last_word
            let mask = W::MAX >> (W::BITS - bit_len);
            let word = (source[src_first_word] >> src_bit
                | source[src_last_word] << (W::BITS - src_bit))
                & mask;
            dest[dst_first_word] &= !(mask << dst_bit);
            dest[dst_first_word] |= word << dst_bit;
        } else if src_bit == dst_bit {
            // src_first_word != src_last_word && dst_first_word != dst_last_word
            let mask = W::MAX << dst_bit;
            dest[dst_first_word] &= !mask;
            dest[dst_first_word] |= source[src_first_word] & mask;

            dest[(1 + dst_first_word)..dst_last_word]
                .copy_from_slice(&source[(1 + src_first_word)..src_last_word]);

            let residual =
                bit_len - (W::BITS - src_bit) - (dst_last_word - dst_first_word - 1) * W::BITS;
            let mask = W::MAX >> (W::BITS - residual);
            dest[dst_last_word] &= !mask;
            dest[dst_last_word] |= source[src_last_word] & mask;
        } else if src_bit < dst_bit {
            // src_first_word != src_last_word && dst_first_word !=
            // dst_last_word
            let dst_mask = W::MAX << dst_bit;
            let src_mask = W::MAX << src_bit;
            let shift = dst_bit - src_bit;
            dest[dst_first_word] &= !dst_mask;
            dest[dst_first_word] |= (source[src_first_word] & src_mask) << shift;

            let mut word = source[src_first_word] >> (W::BITS - shift);
            for i in 1..dst_last_word - dst_first_word {
                dest[dst_first_word + i] = word | source[src_first_word + i] << shift;
                word = source[src_first_word + i] >> (W::BITS - shift);
            }
            let residual =
                bit_len - (W::BITS - dst_bit) - (dst_last_word - dst_first_word - 1) * W::BITS;
            let mask = W::MAX >> (W::BITS - residual);
            dest[dst_last_word] &= !mask;
            dest[dst_last_word] |= source[src_last_word] & mask;
        } else {
            // src_first_word != src_last_word && dst_first_word !=
            // dst_last_word && src_bit > dst_bit
            let dst_mask = W::MAX << dst_bit;
            let src_mask = W::MAX << src_bit;
            let shift = src_bit - dst_bit;
            dest[dst_first_word] &= !dst_mask;
            dest[dst_first_word] |= (source[src_first_word] & src_mask) >> shift;
            dest[dst_first_word] |= source[src_first_word + 1] << (W::BITS - shift);

            let mut word = source[src_first_word + 1] >> shift;

            for i in 1..dst_last_word - dst_first_word {
                word |= source[src_first_word + i + 1] << (W::BITS - shift);
                dest[dst_first_word + i] = word;
                word = source[src_first_word + i + 1] >> shift;
            }

            word |= source[src_last_word] << (W::BITS - shift);

            let residual =
                bit_len - (W::BITS - dst_bit) - (dst_last_word - dst_first_word - 1) * W::BITS;
            let mask = W::MAX >> (W::BITS - residual);
            dest[dst_last_word] &= !mask;
            dest[dst_last_word] |= word & mask;
        }
    }

    /// This implementation keeps a buffer of `W::BITS` bits for reading and
    /// writing, obtaining a significant speedup with respect to the default
    /// implementation.
    #[inline]
    unsafe fn apply_in_place_unchecked<F>(&mut self, mut f: F)
    where
        F: FnMut(W) -> W,
        Self: BitFieldSlice<W>,
    {
        if self.is_empty() {
            return;
        }
        let bit_width = self.bit_width();
        if bit_width == 0 {
            return;
        }
        let mask = self.mask();
        let number_of_words: usize = self.bits.as_ref().len();
        let last_word_idx = number_of_words.saturating_sub(1);

        let mut write_buffer: W = W::ZERO;
        let mut read_buffer: W = *self.bits.as_ref().get_unchecked(0);

        // specialized case because it's much faster
        if bit_width.is_power_of_two() {
            let mut bits_in_buffer = 0;

            // TODO!: figure out how to simplify
            let mut buffer_limit = (self.len() * bit_width) % W::BITS;
            if buffer_limit == 0 {
                buffer_limit = W::BITS;
            }

            for read_idx in 1..number_of_words {
                // pre-load the next word so it loads while we parse the buffer
                let next_word: W = *self.bits.as_ref().get_unchecked(read_idx);

                // parse as much as we can from the buffer
                loop {
                    let next_bits_in_buffer = bits_in_buffer + bit_width;

                    if next_bits_in_buffer > W::BITS {
                        break;
                    }

                    let value = read_buffer & mask;
                    // throw away the bits we just read
                    read_buffer >>= bit_width;
                    // apply user func
                    let new_value = f(value);
                    // put the new value in the write buffer
                    write_buffer |= new_value << bits_in_buffer;

                    bits_in_buffer = next_bits_in_buffer;
                }

                invariant_eq!(read_buffer, W::ZERO);
                *self.bits.as_mut().get_unchecked_mut(read_idx - 1) = write_buffer;
                read_buffer = next_word;
                write_buffer = W::ZERO;
                bits_in_buffer = 0;
            }

            // write the last word if we have some bits left
            while bits_in_buffer < buffer_limit {
                let value = read_buffer & mask;
                // throw away the bits we just read
                read_buffer >>= bit_width;
                // apply user func
                let new_value = f(value);
                // put the new value in the write buffer
                write_buffer |= new_value << bits_in_buffer;
                // -= bit_width but with no casts
                bits_in_buffer += bit_width;
            }

            *self.bits.as_mut().get_unchecked_mut(last_word_idx) = write_buffer;
            return;
        }

        // The position inside the word. In most parametrization of the
        // vector, since the bit_width is not necessarily a integer
        // divisor of the word size, we need to keep track of the position
        // inside the word. As we scroll through the bits, due to the bits
        // remainder, we may need to operate on two words at the same time.
        let mut global_bit_index: usize = 0;

        // The number of words in the bitvec.
        let mut lower_word_limit = 0;
        let mut upper_word_limit = W::BITS;

        // We iterate across the words
        for word_number in 0..last_word_idx {
            // We iterate across the elements in the word.
            while global_bit_index + bit_width <= upper_word_limit {
                // We retrieve the value from the current word.
                let offset = global_bit_index - lower_word_limit;
                global_bit_index += bit_width;
                let element = self.mask() & (read_buffer >> offset);

                // We apply the function to the element.
                let new_element = f(element);

                // We set the element in the new word.
                write_buffer |= new_element << offset;
            }

            // We retrieve the next word from the bitvec.
            let next_word = *self.bits.as_ref().get_unchecked(word_number + 1);

            let mut new_write_buffer = W::ZERO;
            if upper_word_limit != global_bit_index {
                let remainder = upper_word_limit - global_bit_index;
                let offset = global_bit_index - lower_word_limit;
                // We compose the element from the remaining elements in the
                // current word and the elements in the next word.
                let element = ((read_buffer >> offset) | (next_word << remainder)) & self.mask();
                global_bit_index += bit_width;

                // We apply the function to the element.
                let new_element = f(element);

                write_buffer |= new_element << offset;

                new_write_buffer = new_element >> remainder;
            };

            read_buffer = next_word;

            *self.bits.as_mut().get_unchecked_mut(word_number) = write_buffer;

            write_buffer = new_write_buffer;
            lower_word_limit = upper_word_limit;
            upper_word_limit += W::BITS;
        }

        let mut offset = global_bit_index - lower_word_limit;

        // We iterate across the elements in the word.
        while offset < self.len() * bit_width - global_bit_index {
            // We retrieve the value from the current word.
            let element = self.mask() & (read_buffer >> offset);

            // We apply the function to the element.
            let new_element = f(element);

            // We set the element in the new word.
            write_buffer |= new_element << offset;
            offset += bit_width;
        }

        *self.bits.as_mut().get_unchecked_mut(last_word_idx) = write_buffer;
    }
}

impl<W: Word> core::iter::Extend<W> for BitFieldVec<W, Vec<W>> {
    /// Add values from
    fn extend<T: IntoIterator<Item = W>>(&mut self, iter: T) {
        for value in iter {
            self.push(value);
        }
    }
}

/// Equality between bit-field vectors requires that the word is the same, the
/// bit width is the same, and the content is the same.
impl<W: Word, B: AsRef<[W]>, C: AsRef<[W]>> PartialEq<BitFieldVec<W, C>> for BitFieldVec<W, B> {
    fn eq(&self, other: &BitFieldVec<W, C>) -> bool {
        if self.bit_width() != other.bit_width() {
            return false;
        }
        if self.len() != other.len() {
            return false;
        }
        let bit_len = self.len() * self.bit_width();
        if self.bits.as_ref()[..bit_len / W::BITS] != other.bits.as_ref()[..bit_len / W::BITS] {
            return false;
        }

        let residual = bit_len % W::BITS;
        residual == 0
            || (self.bits.as_ref()[bit_len / W::BITS] ^ other.bits.as_ref()[bit_len / W::BITS])
                << (W::BITS - residual)
                == W::ZERO
    }
}

impl Eq for BitFieldVec {}

// Support for unchecked iterators

/// An [`UncheckedIterator`] over the values of a [`BitFieldVec`].
#[derive(Debug, Clone, MemDbg, MemSize)]
pub struct BitFieldVectorUncheckedIterator<'a, W, B>
where
    W: Word,
{
    vec: &'a BitFieldVec<W, B>,
    word_index: usize,
    window: W,
    fill: usize,
}

impl<'a, W: Word, B: AsRef<[W]>> BitFieldVectorUncheckedIterator<'a, W, B> {
    fn new(vec: &'a BitFieldVec<W, B>, index: usize) -> Self {
        if index > vec.len() {
            panic!("Start index out of bounds: {} > {}", index, vec.len());
        }
        let (fill, word_index);
        let window = if index == vec.len() {
            word_index = 0;
            fill = 0;
            W::ZERO
        } else {
            let bit_offset = index * vec.bit_width;
            let bit_index = bit_offset % W::BITS;

            word_index = bit_offset / W::BITS;
            fill = W::BITS - bit_index;
            unsafe {
                // SAFETY: index has been check at the start and it is within bounds
                *vec.bits.as_ref().get_unchecked(word_index) >> bit_index
            }
        };
        Self {
            vec,
            word_index,
            window,
            fill,
        }
    }
}

impl<W: Word, B: AsRef<[W]>> crate::traits::UncheckedIterator
    for BitFieldVectorUncheckedIterator<'_, W, B>
{
    type Item = W;
    unsafe fn next_unchecked(&mut self) -> W {
        let bit_width = self.vec.bit_width;

        if self.fill >= bit_width {
            self.fill -= bit_width;
            let res = self.window & self.vec.mask;
            self.window >>= bit_width;
            return res;
        }

        let res = self.window;
        self.word_index += 1;
        self.window = *self.vec.bits.as_ref().get_unchecked(self.word_index);
        let res = (res | (self.window << self.fill)) & self.vec.mask;
        let used = bit_width - self.fill;
        self.window >>= used;
        self.fill = W::BITS - used;
        res
    }
}

impl<'a, W: Word, B: AsRef<[W]>> IntoUncheckedIterator for &'a BitFieldVec<W, B> {
    type Item = W;
    type IntoUncheckedIter = BitFieldVectorUncheckedIterator<'a, W, B>;
    fn into_unchecked_iter_from(self, from: usize) -> Self::IntoUncheckedIter {
        BitFieldVectorUncheckedIterator::new(self, from)
    }
}

/// An [`UncheckedIterator`] moving backwards over the values of a [`BitFieldVec`].
#[derive(Debug, Clone, MemDbg, MemSize)]
pub struct BitFieldVectorReverseUncheckedIterator<'a, W: Word, B> {
    vec: &'a BitFieldVec<W, B>,
    word_index: usize,
    window: W,
    fill: usize,
}

impl<'a, W: Word, B: AsRef<[W]>> BitFieldVectorReverseUncheckedIterator<'a, W, B> {
    fn new(vec: &'a BitFieldVec<W, B>, index: usize) -> Self {
        if index > vec.len() {
            panic!("Start index out of bounds: {} > {}", index, vec.len());
        }
        let (word_index, fill);

        let window = if index == 0 {
            word_index = 0;
            fill = 0;
            W::ZERO
        } else {
            // We have to handle the case of zero bit width
            let bit_offset = (index * vec.bit_width).saturating_sub(1);
            let bit_index = bit_offset % W::BITS;

            word_index = bit_offset / W::BITS;
            fill = bit_index + 1;
            unsafe {
                // SAFETY: index has been check at the start and it is within bounds
                *vec.bits.as_ref().get_unchecked(word_index) << (W::BITS - fill)
            }
        };
        Self {
            vec,
            word_index,
            window,
            fill,
        }
    }
}

impl<W: Word, B: AsRef<[W]>> crate::traits::UncheckedIterator
    for BitFieldVectorReverseUncheckedIterator<'_, W, B>
{
    type Item = W;
    unsafe fn next_unchecked(&mut self) -> W {
        let bit_width = self.vec.bit_width;

        if self.fill >= bit_width {
            self.fill -= bit_width;
            self.window = self.window.rotate_left(bit_width as u32);
            return self.window & self.vec.mask;
        }

        let mut res = self.window.rotate_left(self.fill as u32);
        self.word_index -= 1;
        self.window = *self.vec.bits.as_ref().get_unchecked(self.word_index);
        let used = bit_width - self.fill;
        res = ((res << used) | (self.window >> (W::BITS - used))) & self.vec.mask;
        self.window <<= used;
        self.fill = W::BITS - used;
        res
    }
}

impl<'a, W: Word, B: AsRef<[W]>> IntoReverseUncheckedIterator for &'a BitFieldVec<W, B> {
    type Item = W;
    type IntoRevUncheckedIter = BitFieldVectorReverseUncheckedIterator<'a, W, B>;

    fn into_rev_unchecked_iter(self) -> Self::IntoRevUncheckedIter {
        BitFieldVectorReverseUncheckedIterator::new(self, self.len())
    }

    fn into_rev_unchecked_iter_from(self, from: usize) -> Self::IntoRevUncheckedIter {
        BitFieldVectorReverseUncheckedIterator::new(self, from)
    }
}

/// An [`Iterator`] over the values of a [`BitFieldVec`].
#[derive(Debug, Clone, MemDbg, MemSize)]
pub struct BitFieldVecIterator<'a, W, B>
where
    W: Word,
{
    unchecked: BitFieldVectorUncheckedIterator<'a, W, B>,
    index: usize,
}

impl<'a, W: Word, B: AsRef<[W]>> BitFieldVecIterator<'a, W, B> {
    fn new(vec: &'a BitFieldVec<W, B>, from: usize) -> Self {
        if from > vec.len() {
            panic!("Start index out of bounds: {} > {}", from, vec.len());
        }
        Self {
            unchecked: BitFieldVectorUncheckedIterator::new(vec, from),
            index: from,
        }
    }
}

impl<W: Word, B: AsRef<[W]>> Iterator for BitFieldVecIterator<'_, W, B> {
    type Item = W;
    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.unchecked.vec.len() {
            // SAFETY: index has just been checked.
            let res = unsafe { self.unchecked.next_unchecked() };
            self.index += 1;
            Some(res)
        } else {
            None
        }
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }
}

impl<W: Word, B: AsRef<[W]>> ExactSizeIterator for BitFieldVecIterator<'_, W, B> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.unchecked.vec.len() - self.index
    }
}

impl<'a, W: Word, B: AsRef<[W]>> IntoIterator for &'a BitFieldVec<W, B> {
    type Item = W;
    type IntoIter = BitFieldVecIterator<'a, W, B>;

    fn into_iter(self) -> Self::IntoIter {
        BitFieldVecIterator::new(self, 0)
    }
}

impl<W: Word, B: AsRef<[W]>> BitFieldVec<W, B> {
    pub fn iter_from(&self, from: usize) -> BitFieldVecIterator<W, B> {
        BitFieldVecIterator::new(self, from)
    }

    pub fn iter(&self) -> BitFieldVecIterator<W, B> {
        self.iter_from(0)
    }
}
/// A tentatively thread-safe vector of bit fields of fixed width.
///
/// This implementation provides some concurrency guarantees, albeit not
/// full-fledged thread safety: more precisely, we can guarantee thread-safety
/// if the bit width is a power of two; otherwise, concurrent writes to values
/// that cross word boundaries might end up in different threads succeding in
/// writing only part of a value. If the user can guarantee that no two threads
/// ever write to the same boundary-crossing value, then no race condition can
/// happen.
///
/// Note that the trait
/// [`AtomicHelper`](crate::traits::bit_field_slice::AtomicHelper) can be used
/// to provide a more convenient naming for some methods.

#[derive(Epserde, Debug, Clone, Hash, MemDbg, MemSize)]
pub struct AtomicBitFieldVec<W: Word + IntoAtomic = usize, B = Vec<<W as IntoAtomic>::AtomicType>> {
    /// The underlying storage.
    bits: B,
    /// The bit width of the values stored in the vector.
    bit_width: usize,
    /// A mask with its lowest `bit_width` bits set to one.
    mask: W,
    /// The length of the vector.
    len: usize,
}

impl<W: Word + IntoAtomic, B> AtomicBitFieldVec<W, B> {
    /// # Safety
    /// `len` * `bit_width` must be between 0 (included) the number of
    /// bits in `bits` (included).
    #[inline(always)]
    pub unsafe fn from_raw_parts(bits: B, bit_width: usize, len: usize) -> Self {
        Self {
            bits,
            bit_width,
            mask: mask(bit_width),
            len,
        }
    }

    #[inline(always)]
    pub fn into_raw_parts(self) -> (B, usize, usize) {
        (self.bits, self.bit_width, self.len)
    }

    /// Returns the mask used to extract values from the vector.
    /// This will keep the lowest `bit_width` bits.
    pub fn mask(&self) -> W {
        self.mask
    }
}

impl<W: Word + IntoAtomic, B: AsRef<[W::AtomicType]>> AtomicBitFieldVec<W, B> {
    /// Returns the backend of the `AtomicBitFieldVec` as a slice of `A`, where `A` is the
    /// atomic variant of `W`.
    pub fn as_slice(&self) -> &[W::AtomicType] {
        self.bits.as_ref()
    }
}

impl<W: Word + IntoAtomic> AtomicBitFieldVec<W>
where
    W::AtomicType: AtomicUnsignedInt,
{
    pub fn new(bit_width: usize, len: usize) -> AtomicBitFieldVec<W> {
        // we need at least two words to avoid branches in the gets
        let n_of_words = Ord::max(1, (len * bit_width).div_ceil(W::BITS));
        AtomicBitFieldVec::<W> {
            bits: (0..n_of_words)
                .map(|_| W::AtomicType::new(W::ZERO))
                .collect(),
            bit_width,
            mask: mask(bit_width),
            len,
        }
    }
}

impl<W: Word + IntoAtomic, B: AsRef<[W::AtomicType]>> AtomicBitFieldVec<W, B>
where
    W::AtomicType: AtomicUnsignedInt + AsBytes,
{
    /// Writes zeros in all values.
    #[deprecated(since = "0.4.4", note = "reset is deprecated in favor of reset_atomic")]
    pub fn reset(&mut self, ordering: Ordering) {
        self.reset_atomic(ordering)
    }
}

impl<W: Word + IntoAtomic, B> BitFieldSliceCore<W::AtomicType> for AtomicBitFieldVec<W, B> {
    #[inline(always)]
    fn bit_width(&self) -> usize {
        debug_assert!(self.bit_width <= W::BITS);
        self.bit_width
    }

    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }
}

impl<W: Word + IntoAtomic, T: AsRef<[W::AtomicType]>> AtomicBitFieldSlice<W>
    for AtomicBitFieldVec<W, T>
where
    W::AtomicType: AtomicUnsignedInt + AsBytes,
{
    #[inline]
    unsafe fn get_atomic_unchecked(&self, index: usize, order: Ordering) -> W {
        let pos = index * self.bit_width;
        let word_index = pos / W::BITS;
        let bit_index = pos % W::BITS;
        let bits = self.bits.as_ref();

        if bit_index + self.bit_width <= W::BITS {
            (bits.get_unchecked(word_index).load(order) >> bit_index) & self.mask
        } else {
            ((bits.get_unchecked(word_index).load(order) >> bit_index)
                | (bits.get_unchecked(word_index + 1).load(order) << (W::BITS - bit_index)))
                & self.mask
        }
    }

    // We reimplement set as we have the mask in the structure.

    /// Sets the element of the slice at the specified index.
    ///
    /// May panic if the index is not in in [0..[len](`BitFieldSliceCore::len`))
    /// or the value does not fit in [`BitFieldSliceCore::bit_width`] bits.
    #[inline(always)]
    fn set_atomic(&self, index: usize, value: W, order: Ordering) {
        panic_if_out_of_bounds!(index, self.len);
        panic_if_value!(value, self.mask, self.bit_width);
        unsafe {
            self.set_atomic_unchecked(index, value, order);
        }
    }

    #[inline]
    unsafe fn set_atomic_unchecked(&self, index: usize, value: W, order: Ordering) {
        debug_assert!(self.bit_width != W::BITS);
        let pos = index * self.bit_width;
        let word_index = pos / W::BITS;
        let bit_index = pos % W::BITS;
        let bits = self.bits.as_ref();

        if bit_index + self.bit_width <= W::BITS {
            // this is consistent
            let mut current = bits.get_unchecked(word_index).load(order);
            loop {
                let mut new = current;
                new &= !(self.mask << bit_index);
                new |= value << bit_index;

                match bits
                    .get_unchecked(word_index)
                    .compare_exchange(current, new, order, order)
                {
                    Ok(_) => break,
                    Err(e) => current = e,
                }
            }
        } else {
            let mut word = bits.get_unchecked(word_index).load(order);
            // try to wait for the other thread to finish
            fence(Ordering::Acquire);
            loop {
                let mut new = word;
                new &= (W::ONE << bit_index) - W::ONE;
                new |= value << bit_index;

                match bits
                    .get_unchecked(word_index)
                    .compare_exchange(word, new, order, order)
                {
                    Ok(_) => break,
                    Err(e) => word = e,
                }
            }
            fence(Ordering::Release);

            // ensures that the compiler does not reorder the two atomic operations
            // this should increase the probability of having consistency
            // between two concurrent writes as they will both execute the set
            // of the bits in the same order, and the release / acquire fence
            // should try to syncronize the threads as much as possible
            compiler_fence(Ordering::SeqCst);

            let mut word = bits.get_unchecked(word_index + 1).load(order);
            fence(Ordering::Acquire);
            loop {
                let mut new = word;
                new &= !(self.mask >> (W::BITS - bit_index));
                new |= value >> (W::BITS - bit_index);

                match bits
                    .get_unchecked(word_index + 1)
                    .compare_exchange(word, new, order, order)
                {
                    Ok(_) => break,
                    Err(e) => word = e,
                }
            }
            fence(Ordering::Release);
        }
    }

    /// Writes zeros in all values.
    fn reset_atomic(&mut self, ordering: Ordering) {
        let bit_len = self.len * self.bit_width;
        let full_words = bit_len / W::BITS;
        let residual = bit_len % W::BITS;
        let bits = self.bits.as_ref();

        #[cfg(feature = "rayon")]
        {
            bits[..full_words]
                .par_iter()
                .for_each(|x| x.store(W::ZERO, ordering));
        }

        #[cfg(not(feature = "rayon"))]
        {
            bits[..full_words]
                .iter()
                .for_each(|x| x.store(W::ZERO, ordering));
        }

        if residual != 0 {
            bits[full_words].fetch_and(W::MAX << residual, ordering);
        }
    }
}

// Conversions

impl<W: Word + IntoAtomic> From<AtomicBitFieldVec<W, Vec<W::AtomicType>>>
    for BitFieldVec<W, Vec<W>>
{
    #[inline]
    fn from(value: AtomicBitFieldVec<W, Vec<W::AtomicType>>) -> Self {
        BitFieldVec {
            bits: unsafe { transmute_vec::<W::AtomicType, W>(value.bits) },
            len: value.len,
            bit_width: value.bit_width,
            mask: value.mask,
        }
    }
}

impl<W: Word + IntoAtomic> From<AtomicBitFieldVec<W, Box<[W::AtomicType]>>>
    for BitFieldVec<W, Box<[W]>>
{
    #[inline]
    fn from(value: AtomicBitFieldVec<W, Box<[W::AtomicType]>>) -> Self {
        BitFieldVec {
            bits: unsafe { transmute_boxed_slice::<W::AtomicType, W>(value.bits) },

            len: value.len,
            bit_width: value.bit_width,
            mask: value.mask,
        }
    }
}

impl<'a, W: Word + IntoAtomic> From<AtomicBitFieldVec<W, &'a [W::AtomicType]>>
    for BitFieldVec<W, &'a [W]>
{
    #[inline]
    fn from(value: AtomicBitFieldVec<W, &'a [W::AtomicType]>) -> Self {
        BitFieldVec {
            bits: unsafe { core::mem::transmute::<&'a [W::AtomicType], &'a [W]>(value.bits) },
            len: value.len,
            bit_width: value.bit_width,
            mask: value.mask,
        }
    }
}

impl<'a, W: Word + IntoAtomic> From<AtomicBitFieldVec<W, &'a mut [W::AtomicType]>>
    for BitFieldVec<W, &'a mut [W]>
{
    #[inline]
    fn from(value: AtomicBitFieldVec<W, &'a mut [W::AtomicType]>) -> Self {
        BitFieldVec {
            bits: unsafe {
                core::mem::transmute::<&'a mut [W::AtomicType], &'a mut [W]>(value.bits)
            },
            len: value.len,
            bit_width: value.bit_width,
            mask: value.mask,
        }
    }
}

impl<W: Word + IntoAtomic> From<BitFieldVec<W, Vec<W>>>
    for AtomicBitFieldVec<W, Vec<W::AtomicType>>
{
    #[inline]
    fn from(value: BitFieldVec<W, Vec<W>>) -> Self {
        AtomicBitFieldVec {
            bits: unsafe { transmute_vec::<W, W::AtomicType>(value.bits) },
            len: value.len,
            bit_width: value.bit_width,
            mask: value.mask,
        }
    }
}

impl<W: Word + IntoAtomic> From<BitFieldVec<W, Box<[W]>>>
    for AtomicBitFieldVec<W, Box<[W::AtomicType]>>
{
    #[inline]
    fn from(value: BitFieldVec<W, Box<[W]>>) -> Self {
        AtomicBitFieldVec {
            bits: unsafe { transmute_boxed_slice::<W, W::AtomicType>(value.bits) },
            len: value.len,
            bit_width: value.bit_width,
            mask: value.mask,
        }
    }
}

impl<'a, W: Word + IntoAtomic> From<BitFieldVec<W, &'a [W]>>
    for AtomicBitFieldVec<W, &'a [W::AtomicType]>
{
    #[inline]
    fn from(value: BitFieldVec<W, &'a [W]>) -> Self {
        AtomicBitFieldVec {
            bits: unsafe { core::mem::transmute::<&'a [W], &'a [W::AtomicType]>(value.bits) },
            len: value.len,
            bit_width: value.bit_width,
            mask: value.mask,
        }
    }
}

impl<'a, W: Word + IntoAtomic> From<BitFieldVec<W, &'a mut [W]>>
    for AtomicBitFieldVec<W, &'a mut [W::AtomicType]>
{
    #[inline]
    fn from(value: BitFieldVec<W, &'a mut [W]>) -> Self {
        AtomicBitFieldVec {
            bits: unsafe {
                core::mem::transmute::<&'a mut [W], &'a mut [W::AtomicType]>(value.bits)
            },
            len: value.len,
            bit_width: value.bit_width,
            mask: value.mask,
        }
    }
}

impl<W: Word> From<BitFieldVec<W, Vec<W>>> for BitFieldVec<W, Box<[W]>> {
    fn from(value: BitFieldVec<W, Vec<W>>) -> Self {
        BitFieldVec {
            bits: value.bits.into_boxed_slice(),
            len: value.len,
            bit_width: value.bit_width,
            mask: value.mask,
        }
    }
}

impl<W: Word> From<BitFieldVec<W, Box<[W]>>> for BitFieldVec<W, Vec<W>> {
    fn from(value: BitFieldVec<W, Box<[W]>>) -> Self {
        BitFieldVec {
            bits: value.bits.into_vec(),
            len: value.len,
            bit_width: value.bit_width,
            mask: value.mask,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_with_capacity() {
        let mut b = BitFieldVec::<usize, _>::with_capacity(10, 100);
        let capacity = b.bits.capacity();
        for _ in 0..100 {
            b.push(0);
        }
        assert_eq!(b.bits.capacity(), capacity);
    }

    fn copy<W: Word, B: AsRef<[W]>, C: AsRef<[W]> + AsMut<[W]>>(
        source: &BitFieldVec<W, B>,
        from: usize,
        dest: &mut BitFieldVec<W, C>,
        to: usize,
        len: usize,
    ) {
        let len = Ord::min(Ord::min(len, dest.len - to), source.len - from);
        for i in 0..len {
            dest.set(to + i, source.get(from + i));
        }
    }

    #[test]
    fn test_copy() {
        for src_pattern in 0..8 {
            for dst_pattern in 0..8 {
                // if from_first_word == from_last_word && to_first_word == to_last_word
                let source = bit_field_vec![3 => src_pattern; 100];
                let mut dest_actual = bit_field_vec![3 => dst_pattern; 100];
                let mut dest_expected = dest_actual.clone();
                source.copy(1, &mut dest_actual, 2, 10);
                copy(&source, 1, &mut dest_expected, 2, 10);
                assert_eq!(dest_actual, dest_expected);
                // else if from_first_word == from_last_word
                let source = bit_field_vec![3 => src_pattern; 100];
                let mut dest_actual = bit_field_vec![3 => dst_pattern; 100];
                let mut dest_expected = dest_actual.clone();
                source.copy(1, &mut dest_actual, 20, 10);
                copy(&source, 1, &mut dest_expected, 20, 10);
                assert_eq!(dest_actual, dest_expected);
                // else if to_first_word == to_last_word
                let source = bit_field_vec![3 => src_pattern; 100];
                let mut dest_actual = bit_field_vec![3 => dst_pattern; 100];
                let mut dest_expected = dest_actual.clone();
                source.copy(20, &mut dest_actual, 1, 10);
                copy(&source, 20, &mut dest_expected, 1, 10);
                assert_eq!(dest_actual, dest_expected);
                // else if src_bit == dest_bit (residual = 1)
                let source = bit_field_vec![3 => src_pattern; 1000];
                let mut dest_actual = bit_field_vec![3 => dst_pattern; 1000];
                let mut dest_expected = dest_actual.clone();
                source.copy(3, &mut dest_actual, 3 + 3 * 128, 40);
                copy(&source, 3, &mut dest_expected, 3 + 3 * 128, 40);
                assert_eq!(dest_actual, dest_expected);
                // else if src_bit == dest_bit (residual = 64)
                let source = bit_field_vec![3 => src_pattern; 1000];
                let mut dest_actual = bit_field_vec![3 => dst_pattern; 1000];
                let mut dest_expected = dest_actual.clone();
                source.copy(3, &mut dest_actual, 3 + 3 * 128, 61);
                copy(&source, 3, &mut dest_expected, 3 + 3 * 128, 61);
                assert_eq!(dest_actual, dest_expected);
                // else if src_bit < dest_bit (residual = 1)
                let source = bit_field_vec![3 => src_pattern; 1000];
                let mut dest_actual = bit_field_vec![3 => dst_pattern; 1000];
                let mut dest_expected = dest_actual.clone();
                source.copy(3, &mut dest_actual, 7 + 64 * 3, 40);
                copy(&source, 3, &mut dest_expected, 7 + 64 * 3, 40);
                assert_eq!(dest_actual, dest_expected);
                // else if src_bit < dest_bit (residual = 64)
                let source = bit_field_vec![3 => src_pattern; 1000];
                let mut dest_actual = bit_field_vec![3 => dst_pattern; 1000];
                let mut dest_expected = dest_actual.clone();
                source.copy(3, &mut dest_actual, 7 + 64 * 3, 40 + 17);
                copy(&source, 3, &mut dest_expected, 7 + 64 * 3, 40 + 17);
                assert_eq!(dest_actual, dest_expected);
                // else if src_bit > dest_bit (residual = 1)
                let source = bit_field_vec![3 => src_pattern; 1000];
                let mut dest_actual = bit_field_vec![3 => dst_pattern; 1000];
                let mut dest_expected = dest_actual.clone();
                source.copy(7, &mut dest_actual, 3 + 64 * 3, 40 + 64);
                copy(&source, 7, &mut dest_expected, 3 + 64 * 3, 40 + 64);
                assert_eq!(dest_actual, dest_expected);
                // else if src_bit > dest_bit (residual = 64)
                let source = bit_field_vec![3 => src_pattern; 1000];
                let mut dest_actual = bit_field_vec![3 => dst_pattern; 1000];
                let mut dest_expected = dest_actual.clone();
                source.copy(7, &mut dest_actual, 3 + 64 * 3, 40 + 21 + 64);
                copy(&source, 7, &mut dest_expected, 3 + 64 * 3, 40 + 21 + 64);
                assert_eq!(dest_actual, dest_expected);
            }
        }
    }
}
