/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Operations on bit vectors.
//!
//! `sux` does not provide a dedicated trait for bit vectors (whereas it
//! provides a trait for [bit-field slices]).
//! Rather, it considers anything that is `AsRef<[W]>` (where `W` implements
//! [`Word`]) and implements [`BitLength`] as a bit vector.
//!
//! This approach was chosen for efficiency reasons: all methods are implemented
//! as efficiently as possible for a concrete representation. Other crates opt
//! for a general trait representing a bit vector, but in our experiments the
//! resulting code was not as efficient as the code we obtain.
//!
//! The Rust type system makes the approach quite flexible. We cannot, however,
//! accommodate implicit representation of bit vectors (e.g., compressed or
//! algorithmic). This is in fact in line with Rust's philosophy: the
//! [`Index`] trait returns a reference, which forces an
//! explicit representation of sequences (an alternative approach is provided by
//! the [`value-traits`] crate, which is used by bit-field slices).
//!
//! All traits provided in this module are extension traits. They have no
//! unimplemented methods: just pulling them into scope will provide anything
//! that is `AsRef<[W]>` and implements [`BitLength`] with the operations of
//! a bit vector.
//!
//! Iteration on the bits of the vector, or on the positions of the ones or of the
//! zeros, is provided by means of structures that can be reused.
//!
//! The reference implementations using these traits are
//! [`BitVec`] and [`AtomicBitVec`].
//!
//! [`value-traits`]: https://crates.io/crates/value-traits
//! [bit-field slices]: crate::traits::bit_field_slice
//! [`BitVec`]: crate::bits::BitVec
//! [`AtomicBitVec`]: crate::bits::AtomicBitVec
//! [`Index`]: std::ops::Index

#[cfg(feature = "rayon")]
use crate::ParallelWithLen;
use crate::{bits::test_unaligned_pos, traits::Word};
use ambassador::delegatable_trait;
use atomic_primitive::PrimitiveAtomicUnsigned;
use impl_tools::autoimpl;
use mem_dbg::{MemDbg, MemSize};
use num_primitive::PrimitiveInteger;
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use std::{iter::FusedIterator, sync::atomic::Ordering};

macro_rules! panic_if_out_of_bounds {
    ($index: expr, $len: expr) => {
        if $index >= $len {
            panic!("Bit index out of bounds: {} >= {}", $index, $len)
        }
    };
}

#[inline(always)]
fn assert_backing_word(index: usize, bits_per_word: usize, backing_words: usize) {
    let word_index = index / bits_per_word;
    assert!(
        word_index < backing_words,
        "Bit-vector backing storage is too short: word {word_index} >= {backing_words}"
    );
}

/// Reads a possibly-unaligned `width`-bit value starting at bit position `pos`
/// from `data`, using a branchless read on little-endian targets.
///
/// # Safety
///
/// - `width + (pos % 8)` must be at most `W::BITS`.
/// - Reading `size_of::<W>()` bytes starting at byte offset `pos / 8` must not
///   exceed `data`.
#[inline(always)]
unsafe fn read_unaligned_value<W: Word>(data: &[W], pos: usize, width: usize) -> W {
    if width == 0 {
        return W::ZERO;
    }
    #[cfg(target_endian = "big")]
    {
        // Endian-independent word-based extraction; the caller's padding-word
        // guarantee keeps the two-word straddle in bounds.
        let word_bits = usize::try_from(W::BITS).expect("word width fits in usize");
        let word_index = pos / word_bits;
        let bit_index = pos % word_bits;
        let low = data[word_index] >> bit_index;
        let raw = if bit_index + width <= word_bits {
            low
        } else {
            low | (data[word_index + 1] << (word_bits - bit_index))
        };
        let l = word_bits - width;
        return (raw << l) >> l;
    }
    #[cfg(target_endian = "little")]
    {
        debug_assert!(
            pos / 8 + size_of::<W>() <= std::mem::size_of_val(data),
            "unaligned read at bit position {} would exceed allocation",
            pos,
        );
        let base_ptr = data.as_ptr() as *const u8;
        // SAFETY: the caller guarantees the size_of::<W>()-byte read at byte
        // offset pos/8 stays within `data`; read_unaligned handles misalignment.
        let ptr = unsafe { base_ptr.add(pos / 8) } as *const W;
        let word = unsafe { core::ptr::read_unaligned(ptr) };
        let l = usize::try_from(W::BITS).expect("word width fits in usize") - width;
        ((word >> (pos % 8)) << l) >> l
    }
}

/// Reads a possibly-unaligned full word starting at bit position `pos` from
/// `data`.
///
/// # Safety
///
/// Reading `size_of::<W>()` bytes starting at byte offset `pos / 8` must not
/// exceed `data`.
#[inline(always)]
unsafe fn read_unaligned_word<W: Word>(data: &[W], pos: usize) -> W {
    #[cfg(target_endian = "big")]
    {
        // Endian-independent word-based read; the caller's padding-word
        // guarantee keeps the two-word straddle in bounds.
        let word_bits = usize::try_from(W::BITS).expect("word width fits in usize");
        let word_index = pos / word_bits;
        let bit_index = pos % word_bits;
        return if bit_index == 0 {
            data[word_index]
        } else {
            (data[word_index] >> bit_index) | (data[word_index + 1] << (word_bits - bit_index))
        };
    }
    #[cfg(target_endian = "little")]
    {
        debug_assert!(
            pos / 8 + size_of::<W>() <= std::mem::size_of_val(data),
            "unaligned read at bit position {} would exceed allocation",
            pos,
        );
        let base_ptr = data.as_ptr() as *const u8;
        // SAFETY: the caller guarantees the size_of::<W>()-byte read at byte
        // offset pos/8 stays within `data`; read_unaligned handles misalignment.
        let ptr = unsafe { base_ptr.add(pos / 8) } as *const W;
        unsafe { core::ptr::read_unaligned(ptr) >> (pos % 8) }
    }
}

/// A trait expressing a length in bits.
///
/// This trait is typically used in conjunction with
/// [`AsRef<[W]>`](core::convert::AsRef) to provide word-based access to a bit
/// vector on words of type `W`.
#[autoimpl(for<T: trait + ?Sized> &T, &mut T, Box<T>)]
#[delegatable_trait(inline = "always")]
pub trait BitLength {
    /// Returns a length in bits.
    fn len(&self) -> usize;

    /// Returns true if the length is zero.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<W: Word, T: ?Sized + AsRef<[W]> + BitLength> BitVecOps<W> for T {}

/// Read-only operations on bit vectors.
pub trait BitVecOps<W: Word>: AsRef<[W]> + BitLength {
    /// Returns true if the bit of given index is set.
    #[inline]
    fn get(&self, index: usize) -> bool {
        panic_if_out_of_bounds!(index, self.len());
        let bits_per_word = usize::try_from(W::BITS).expect("word width fits in usize");
        // Bind the backing slice once so the bounds check and the read use the
        // same slice even if a backend's `AsRef` is not stable across calls.
        let bits = self.as_ref();
        assert_backing_word(index, bits_per_word, bits.len());
        let word_index = index / bits_per_word;
        // SAFETY: `assert_backing_word` proved `word_index < bits.len()` on this
        // exact slice.
        let word = unsafe { *bits.get_unchecked(word_index) };
        (word >> (index % bits_per_word)) & W::ONE != W::ZERO
    }

    /// Returns true if the bit of given index is set, without
    /// bound checks.
    ///
    /// # Safety
    ///
    /// `index` must be between 0 (included) and [`BitLength::len`] (excluded),
    /// and `self.as_ref()` must contain the word holding `index`.
    #[inline(always)]
    unsafe fn get_unchecked(&self, index: usize) -> bool {
        let bits_per_word = W::BITS as usize;
        let word_index = index / bits_per_word;
        let word = unsafe { *self.as_ref().get_unchecked(word_index) };
        (word >> (index % bits_per_word)) & W::ONE != W::ZERO
    }

    /// Like [`BitVecValueOps::get_bits`], but using a branchless unaligned
    /// read.
    ///
    /// The read loads one full word starting at byte offset `pos / 8`,
    /// shifts right by `pos % 8`, and masks to `width` bits. This
    /// avoids a branch at the cost of a *position-dependent* width
    /// constraint: the read is valid iff `width + (pos % 8) <=
    /// W::BITS` (where `W` is the word type of the backend).
    ///
    /// Note that this is **not** the same constraint used by
    /// [`BitFieldVec::get_unaligned`], which can exploit the fact that
    /// its positions are multiples of `bit_width` to allow looser
    /// widths such as `W::BITS - 4` and `W::BITS`. Here `pos` is
    /// arbitrary, so in the worst case `pos % 8 == 7`, and only widths
    /// up to `W::BITS - 7` are unconditionally safe.
    ///
    /// Additionally, a padding word must be present at the end of the
    /// underlying storage.
    ///
    /// # Panics
    ///
    /// Panics if `pos + width` exceeds the bit length, if
    /// `width + (pos % 8)` exceeds `W::BITS`, or if the read would
    /// exceed the allocation.
    ///
    /// [`BitFieldVec::get_unaligned`]: crate::bits::BitFieldVec::get_unaligned
    fn get_value_unaligned(&self, pos: usize, width: usize) -> W {
        assert!(
            test_unaligned_pos!(W, pos, width),
            "bit width {} at bit position {} does not fit in a single unaligned read on word type {} (width + (pos % 8) must be <= {})",
            width,
            pos,
            stringify!(W),
            W::BITS as usize,
        );
        let end = pos
            .checked_add(width)
            .expect("bit range end (pos + width) overflows usize");
        assert!(
            end <= self.len(),
            "bit range {}..{} out of bounds for length {}",
            pos,
            end,
            self.len()
        );
        let data = self.as_ref();
        assert!(
            pos / 8 + size_of::<W>() <= std::mem::size_of_val(data),
            "unaligned read at bit position {} would exceed allocation",
            pos,
        );
        // SAFETY: `end <= self.len()` (checked above) keeps `pos + width` within
        // the logical length, `test_unaligned_pos!` bounds `width + (pos % 8) <=
        // W::BITS`, and this assertion proves the size_of::<W>()-byte read at byte
        // offset pos/8 stays within `data` -- the exact slice passed to the reader.
        unsafe { read_unaligned_value::<W>(data, pos, width) }
    }

    /// Like [`BitVecValueOps::get_bits_unchecked`], but using a
    /// branchless unaligned read.
    ///
    /// See [`get_value_unaligned`](Self::get_value_unaligned) for the
    /// algorithm and the position-dependent width constraint.
    ///
    /// # Safety
    ///
    /// - `width + (pos % 8)` must be at most `W::BITS`. In particular,
    ///   for *arbitrary* `pos`, only widths up to `W::BITS - 7` are
    ///   unconditionally safe; larger widths (up to `W::BITS`) are
    ///   safe only when `pos` is byte-aligned enough to leave room.
    /// - `pos + width` must not exceed the bit length.
    /// - A padding word must be present at the end of the underlying storage so
    ///   that reading `size_of::<W>()` bytes starting at byte offset `pos / 8`
    ///   does not exceed the allocation.
    #[inline]
    unsafe fn get_value_unaligned_unchecked(&self, pos: usize, width: usize) -> W {
        debug_assert!(
            test_unaligned_pos!(W, pos, width),
            "bit width {} at bit position {} does not fit in a single unaligned read on word type {} (width + (pos % 8) must be <= {})",
            width,
            pos,
            stringify!(W),
            usize::try_from(W::BITS).expect("word width fits in usize"),
        );
        // SAFETY: the caller guarantees `width + (pos % 8) <= W::BITS` and that a
        // padding word makes the size_of::<W>()-byte read at byte offset pos/8
        // stay within the backing slice.
        unsafe { read_unaligned_value::<W>(self.as_ref(), pos, width) }
    }

    /// Return the result of an unaligned read of a full word starting at bit
    /// position `pos` checking that the read does not exceed the allocation.
    ///
    /// The actual number of valid bits in the word is `W::BITS - (pos % 8)`.
    fn get_unaligned(&self, pos: usize) -> W {
        let data = self.as_ref();
        assert!(
            pos / 8 + size_of::<W>() <= std::mem::size_of_val(data),
            "unaligned read at bit position {} would exceed allocation",
            pos,
        );
        // SAFETY: the assertion proves the size_of::<W>()-byte read at byte
        // offset pos/8 stays within `data`, the exact slice passed to the reader.
        unsafe { read_unaligned_word::<W>(data, pos) }
    }

    /// Return the result of an unaligned read of a full word starting at bit
    /// position `pos` without checking that the read does not exceed the
    /// allocation.
    ///
    /// The actual number of valid bits in the word is `W::BITS - (pos % 8)`.
    ///
    /// # Safety
    ///
    /// Reading `size_of::<W>()` bytes starting at byte offset `pos / 8`
    /// must not exceed the allocation.
    #[inline(always)]
    unsafe fn get_unaligned_unchecked(&self, pos: usize) -> W {
        // SAFETY: the caller guarantees the size_of::<W>()-byte read at byte
        // offset pos/8 stays within the backing slice.
        unsafe { read_unaligned_word::<W>(self.as_ref(), pos) }
    }

    /// Returns an iterator over the bits of this bit vector as booleans.
    #[inline(always)]
    fn iter(&self) -> BitIter<'_, W> {
        BitIter::new(self.as_ref(), self.len())
    }

    /// Returns an iterator over the positions of the ones in this bit vector.
    fn iter_ones(&self) -> OnesIter<'_, W> {
        OnesIter::new(self.as_ref(), self.len())
    }

    /// Returns an iterator over the positions of the zeros in this bit vector.
    fn iter_zeros(&self) -> ZerosIter<'_, W> {
        ZerosIter::new(self.as_ref(), self.len())
    }

    /// Returns the number of ones in the bit vector.
    ///
    /// Dirty bits past the logical length are masked out. The computation
    /// scans the backing words; for a constant-time count on structures that
    /// cache it, use [`NumBits::num_ones`](crate::traits::NumBits::num_ones).
    fn count_ones(&self) -> usize {
        let bits_per_word = W::BITS as usize;
        let full_words = self.len() / bits_per_word;
        let residual = self.len() % bits_per_word;
        let bits = self.as_ref();
        let mut num_ones: usize = bits[..full_words]
            .iter()
            .map(|x| x.count_ones() as usize)
            .sum();
        if residual != 0 {
            num_ones += (bits[full_words] << (bits_per_word - residual)).count_ones() as usize;
        }
        num_ones
    }

    /// Returns the number of zeros in the bit vector.
    ///
    /// Dirty bits past the logical length are masked out. See
    /// [`count_ones`](Self::count_ones) for the cost.
    #[inline]
    fn count_zeros(&self) -> usize {
        self.len() - self.count_ones()
    }

    /// A parallel version of [`count_ones`](Self::count_ones).
    #[cfg(feature = "rayon")]
    fn par_count_ones(&self) -> usize {
        let bits_per_word = W::BITS as usize;
        let full_words = self.len() / bits_per_word;
        let residual = self.len() % bits_per_word;
        let bits = self.as_ref();
        let mut num_ones;
        num_ones = bits[..full_words]
            .par_iter()
            .with_len(crate::RAYON_MIN_LEN)
            .map(|x| x.count_ones() as usize)
            .sum();
        if residual != 0 {
            num_ones +=
                (self.as_ref()[full_words] << (bits_per_word - residual)).count_ones() as usize
        }

        num_ones
    }
}

impl<W: Word, T: AsRef<[W]> + AsMut<[W]> + BitLength> BitVecOpsMut<W> for T {}

/// Mutation operations on bit vectors.
pub trait BitVecOpsMut<W: Word>: AsRef<[W]> + AsMut<[W]> + BitLength {
    /// Sets the bit of given index to the given value.
    #[inline]
    fn set(&mut self, index: usize, value: bool) {
        panic_if_out_of_bounds!(index, self.len());
        let bits_per_word = usize::try_from(W::BITS).expect("word width fits in usize");
        // Bind the mutable backing once and check it: the previous code
        // validated `self.as_ref()` but wrote through `self.as_mut()`, which is
        // unsound if a backend's `AsRef`/`AsMut` slices ever disagree in length.
        let bits = self.as_mut();
        assert_backing_word(index, bits_per_word, bits.len());
        let word_index = index / bits_per_word;
        let bit_index = index % bits_per_word;
        // SAFETY: `assert_backing_word` proved `word_index < bits.len()` on this
        // exact mutable slice, so the unchecked writes are in bounds.
        unsafe {
            if value {
                *bits.get_unchecked_mut(word_index) |= W::ONE << bit_index;
            } else {
                *bits.get_unchecked_mut(word_index) &= !(W::ONE << bit_index);
            }
        }
    }

    /// Sets the bit of given index to the given value without bound checks.
    ///
    /// # Safety
    ///
    /// `index` must be between 0 (included) and [`BitLength::len`] (excluded),
    /// and `self.as_mut()` must contain the word holding `index`.
    #[inline(always)]
    unsafe fn set_unchecked(&mut self, index: usize, value: bool) {
        let bits_per_word = W::BITS as usize;
        let word_index = index / bits_per_word;
        let bit_index = index % bits_per_word;
        let bits = self.as_mut();
        // For constant values, this should be inlined with no test.
        unsafe {
            if value {
                *bits.get_unchecked_mut(word_index) |= W::ONE << bit_index;
            } else {
                *bits.get_unchecked_mut(word_index) &= !(W::ONE << bit_index);
            }
        }
    }

    /// Sets all bits to the given value.
    fn fill(&mut self, value: bool) {
        let bits_per_word = W::BITS as usize;
        let full_words = self.len() / bits_per_word;
        let residual = self.len() % bits_per_word;
        let bits = self.as_mut();
        let word_value: W = if value { !W::ZERO } else { W::ZERO };
        bits[..full_words].iter_mut().for_each(|x| *x = word_value);
        if residual != 0 {
            let mask = (W::ONE << residual) - W::ONE;
            bits[full_words] = (bits[full_words] & !mask) | (word_value & mask);
        }
    }

    /// Sets all bits to the given value using a parallel implementation.
    #[cfg(feature = "rayon")]
    fn par_fill(&mut self, value: bool) {
        let bits_per_word = W::BITS as usize;
        let full_words = self.len() / bits_per_word;
        let residual = self.len() % bits_per_word;
        let bits = self.as_mut();
        let word_value: W = if value { !W::ZERO } else { W::ZERO };
        bits[..full_words]
            .par_iter_mut()
            .with_len(crate::RAYON_MIN_LEN)
            .for_each(|x| *x = word_value);
        if residual != 0 {
            let mask = (W::ONE << residual) - W::ONE;
            bits[full_words] = (bits[full_words] & !mask) | (word_value & mask);
        }
    }

    /// Sets all bits to zero.
    fn reset(&mut self) {
        self.fill(false);
    }

    /// Sets all bits to zero using a parallel implementation.
    #[cfg(feature = "rayon")]
    fn par_reset(&mut self) {
        self.par_fill(false);
    }

    /// Flip all bits.
    fn flip(&mut self) {
        let bits_per_word = W::BITS as usize;
        let full_words = self.len() / bits_per_word;
        let residual = self.len() % bits_per_word;
        let bits = self.as_mut();
        bits[..full_words].iter_mut().for_each(|x| *x = !*x);
        if residual != 0 {
            let mask = (W::ONE << residual) - W::ONE;
            bits[full_words] = (bits[full_words] & !mask) | (!bits[full_words] & mask);
        }
    }

    /// Flips all bits using a parallel implementation.
    #[cfg(feature = "rayon")]
    fn par_flip(&mut self) {
        let bits_per_word = W::BITS as usize;
        let full_words = self.len() / bits_per_word;
        let residual = self.len() % bits_per_word;
        let bits = self.as_mut();
        bits[..full_words]
            .par_iter_mut()
            .with_len(crate::RAYON_MIN_LEN)
            .for_each(|x| *x = !*x);
        if residual != 0 {
            let mask = (W::ONE << residual) - W::ONE;
            bits[full_words] = (bits[full_words] & !mask) | (!bits[full_words] & mask);
        }
    }
}

/// Operations for reading multi-bit values from a bit vector at arbitrary
/// bit positions.
///
/// Unlike [`BitVecOps`] and [`BitVecOpsMut`], this trait does not have a
/// blanket implementation, allowing different types to provide specialized
/// implementations (e.g., using unaligned reads).
pub trait BitVecValueOps<W: Word> {
    /// Reads `width` bits starting at bit position `pos`.
    ///
    /// # Panics
    ///
    /// Panics if `pos + width` exceeds the bit length or if `width` >
    /// `W::BITS`.
    fn get_bits(&self, pos: usize, width: usize) -> W;

    /// Reads `width` bits starting at bit position `pos`, without bounds
    /// checks.
    ///
    /// # Safety
    ///
    /// - `pos + width` must not exceed the bit length of the underlying
    ///   storage.
    /// - `width` must be at most `W::BITS`.
    unsafe fn get_bits_unchecked(&self, pos: usize, width: usize) -> W;
}

/// An iterator over the bits of a bit vector as booleans.
#[derive(Debug, Clone, MemSize, MemDbg)]
pub struct BitIter<'a, W: Word> {
    bits: &'a [W],
    len: usize,
    next_bit_pos: usize,
}

impl<'a, W: Word> BitIter<'a, W> {
    /// Creates an iterator over the first `len` bits in `bits`.
    ///
    /// The backing slice is snapshotted once, so iteration is sound even if the
    /// `AsRef` implementation does not return the same slice on every call.
    ///
    /// # Panics
    ///
    /// Panics if the backing slice cannot hold `len` bits.
    pub fn new<B: ?Sized + AsRef<[W]>>(bits: &'a B, len: usize) -> Self {
        let bits = bits.as_ref();
        let bits_per_word = usize::try_from(W::BITS).expect("word width fits in usize");
        assert!(
            len.div_ceil(bits_per_word) <= bits.len(),
            "Bit-vector backing storage is too short for {len} bits"
        );
        BitIter {
            bits,
            len,
            next_bit_pos: 0,
        }
    }
}

impl<W: Word> Iterator for BitIter<'_, W> {
    type Item = bool;
    fn next(&mut self) -> Option<bool> {
        if self.next_bit_pos == self.len {
            return None;
        }
        let bits_per_word = usize::try_from(W::BITS).expect("word width fits in usize");
        let word_idx = self.next_bit_pos / bits_per_word;
        let bit_idx = self.next_bit_pos % bits_per_word;
        // SAFETY: `new` checked `len.div_ceil(bits_per_word) <= self.bits.len()`
        // against this exact snapshotted slice, and `next_bit_pos < len` here, so
        // `word_idx < self.bits.len()`.
        let word = unsafe { *self.bits.get_unchecked(word_idx) };
        let bit = (word >> bit_idx) & W::ONE;
        self.next_bit_pos += 1;
        Some(bit != W::ZERO)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let rem = self.len - self.next_bit_pos;
        (rem, Some(rem))
    }
}

impl<W: Word> ExactSizeIterator for BitIter<'_, W> {
    fn len(&self) -> usize {
        self.len - self.next_bit_pos
    }
}

impl<W: Word> FusedIterator for BitIter<'_, W> {}

/// An iterator over the positions of the ones in a bit vector.
#[derive(Debug, Clone, MemSize, MemDbg)]
pub struct OnesIter<'a, W: Word> {
    bits: &'a [W],
    len: usize,
    word_idx: usize,
    word: W,
}

impl<'a, W: Word> OnesIter<'a, W> {
    pub fn new<B: ?Sized + AsRef<[W]>>(bits: &'a B, len: usize) -> Self {
        let bits = bits.as_ref();
        let bits_per_word = usize::try_from(W::BITS).expect("word width fits in usize");
        debug_assert!(len <= bits.len() * bits_per_word);
        let word = if bits.is_empty() {
            W::ZERO
        } else {
            // SAFETY: the slice is non-empty, so index 0 is in bounds.
            unsafe { *bits.get_unchecked(0) }
        };
        Self {
            bits,
            len,
            word_idx: 0,
            word,
        }
    }
}

impl<W: Word> Iterator for OnesIter<'_, W> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        let bits_per_word = usize::try_from(W::BITS).expect("word width fits in usize");
        // find the next word with ones
        while self.word == W::ZERO {
            self.word_idx += 1;
            if self.word_idx >= self.bits.len() {
                return None;
            }
            // SAFETY: the bound check above proved `word_idx < self.bits.len()`.
            self.word = unsafe { *self.bits.get_unchecked(self.word_idx) };
        }
        // find the lowest bit set index in the word
        let bit_idx = usize::try_from(self.word.trailing_zeros()).expect("bit index fits usize");
        // compute the global bit index
        let res = (self.word_idx * bits_per_word) + bit_idx;
        if res >= self.len {
            None
        } else {
            // clear the lowest bit set
            self.word &= self.word - W::ONE;
            Some(res)
        }
    }
}

impl<W: Word> FusedIterator for OnesIter<'_, W> {}

/// An iterator over the positions of the zeros in a bit vector.
#[derive(Debug, Clone, MemSize, MemDbg)]
pub struct ZerosIter<'a, W: Word> {
    bits: &'a [W],
    len: usize,
    word_idx: usize,
    word: W,
}

impl<'a, W: Word> ZerosIter<'a, W> {
    pub fn new<B: ?Sized + AsRef<[W]>>(bits: &'a B, len: usize) -> Self {
        let bits = bits.as_ref();
        let bits_per_word = usize::try_from(W::BITS).expect("word width fits in usize");
        debug_assert!(len <= bits.len() * bits_per_word);
        let word = if bits.is_empty() {
            W::ZERO
        } else {
            // SAFETY: the slice is non-empty, so index 0 is in bounds.
            unsafe { !*bits.get_unchecked(0) }
        };
        Self {
            bits,
            len,
            word_idx: 0,
            word,
        }
    }
}

impl<W: Word> Iterator for ZerosIter<'_, W> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        let bits_per_word = usize::try_from(W::BITS).expect("word width fits in usize");
        // find the next flipped word with zeros
        while self.word == W::ZERO {
            self.word_idx += 1;
            if self.word_idx >= self.bits.len() {
                return None;
            }
            // SAFETY: the bound check above proved `word_idx < self.bits.len()`.
            self.word = unsafe { !*self.bits.get_unchecked(self.word_idx) };
        }
        // find the lowest zero bit index in the word
        let bit_idx = usize::try_from(self.word.trailing_zeros()).expect("bit index fits usize");
        // compute the global bit index
        let res = (self.word_idx * bits_per_word) + bit_idx;
        if res >= self.len {
            None
        } else {
            // clear the lowest bit set
            self.word &= self.word - W::ONE;
            Some(res)
        }
    }
}

impl<W: Word> FusedIterator for ZerosIter<'_, W> {}

impl<A: PrimitiveAtomicUnsigned<Value: Word>, T: ?Sized + AsRef<[A]> + BitLength> AtomicBitVecOps<A>
    for T
{
}

/// Operations on atomic bit vectors.
///
/// Parameterized by the atomic type `A` (e.g., `AtomicU64`), not the word type.
/// This avoids method-resolution ambiguity with [`BitVecOpsMut`], because
/// [`PrimitiveAtomicUnsigned`] is only implemented for atomic types, so the compiler can
/// definitively rule out non-atomic backends.
pub trait AtomicBitVecOps<A: PrimitiveAtomicUnsigned<Value: Word>>: AsRef<[A]> + BitLength {
    /// Returns true if the bit of given index is set.
    ///
    /// This method performs a single atomic operation with the given memory
    /// ordering.
    fn get(&self, index: usize, ordering: Ordering) -> bool {
        panic_if_out_of_bounds!(index, self.len());
        let bits_per_word = usize::try_from(A::Value::BITS).expect("word width fits in usize");
        // Bind the backing slice once so the bounds check and the atomic access
        // use the same slice even if a backend's `AsRef` is not stable.
        let bits = self.as_ref();
        assert_backing_word(index, bits_per_word, bits.len());
        let word_index = index / bits_per_word;
        // SAFETY: `assert_backing_word` proved `word_index < bits.len()` on this
        // exact slice; the load is a single atomic operation.
        let word = unsafe { bits.get_unchecked(word_index).load(ordering) };
        (word >> (index % bits_per_word)) & A::Value::ONE != A::Value::ZERO
    }

    /// Sets the bit of given index to the given value.
    ///
    /// This method performs a single atomic operation with the given memory
    /// ordering.
    fn set(&self, index: usize, value: bool, ordering: Ordering) {
        panic_if_out_of_bounds!(index, self.len());
        let bits_per_word = usize::try_from(A::Value::BITS).expect("word width fits in usize");
        // Bind the backing slice once so the bounds check and the atomic write
        // use the same slice even if a backend's `AsRef` is not stable.
        let bits = self.as_ref();
        assert_backing_word(index, bits_per_word, bits.len());
        let word_index = index / bits_per_word;
        let bit_index = index % bits_per_word;
        // SAFETY: `assert_backing_word` proved `word_index < bits.len()` on this
        // exact slice; each RMW is a single atomic operation.
        unsafe {
            if value {
                bits.get_unchecked(word_index)
                    .fetch_or(A::Value::ONE << bit_index, ordering);
            } else {
                bits.get_unchecked(word_index)
                    .fetch_and(!(A::Value::ONE << bit_index), ordering);
            }
        }
    }

    /// Sets the bit of given index to the given value and returns the previous
    /// value.
    ///
    /// This method performs a single atomic operation with the given memory
    /// ordering.
    fn swap(&self, index: usize, value: bool, ordering: Ordering) -> bool {
        panic_if_out_of_bounds!(index, self.len());
        let bits_per_word = usize::try_from(A::Value::BITS).expect("word width fits in usize");
        // Bind the backing slice once so the bounds check and the atomic RMW use
        // the same slice even if a backend's `AsRef` is not stable.
        let bits = self.as_ref();
        assert_backing_word(index, bits_per_word, bits.len());
        let word_index = index / bits_per_word;
        let bit_index = index % bits_per_word;
        // SAFETY: `assert_backing_word` proved `word_index < bits.len()` on this
        // exact slice; the RMW is a single atomic operation.
        let old_word = unsafe {
            if value {
                bits.get_unchecked(word_index)
                    .fetch_or(A::Value::ONE << bit_index, ordering)
            } else {
                bits.get_unchecked(word_index)
                    .fetch_and(!(A::Value::ONE << bit_index), ordering)
            }
        };
        (old_word >> bit_index) & A::Value::ONE != A::Value::ZERO
    }

    /// Returns true if the bit of given index is set.
    ///
    /// This method performs a single atomic operation with the given memory
    /// ordering.
    ///
    /// # Safety
    ///
    /// `index` must be between 0 (included) and [`BitLength::len`] (excluded),
    /// and `self.as_ref()` must contain the atomic word holding `index`.
    #[inline]
    unsafe fn get_unchecked(&self, index: usize, ordering: Ordering) -> bool {
        let bits_per_word = A::Value::BITS as usize;
        let word_index = index / bits_per_word;
        let bits = self.as_ref();
        let word = unsafe { bits.get_unchecked(word_index).load(ordering) };
        (word >> (index % bits_per_word)) & A::Value::ONE != A::Value::ZERO
    }

    /// Sets the bit of given index to the given value.
    ///
    /// This method performs a single atomic operation with the given memory
    /// ordering.
    ///
    /// # Safety
    ///
    /// `index` must be between 0 (included) and [`BitLength::len`] (excluded),
    /// and `self.as_ref()` must contain the atomic word holding `index`.
    #[inline]
    unsafe fn set_unchecked(&self, index: usize, value: bool, ordering: Ordering) {
        let bits_per_word = A::Value::BITS as usize;
        let word_index = index / bits_per_word;
        let bit_index = index % bits_per_word;
        let bits = self.as_ref();

        // For constant values, this should be inlined with no test.
        unsafe {
            if value {
                bits.get_unchecked(word_index)
                    .fetch_or(A::Value::ONE << bit_index, ordering);
            } else {
                bits.get_unchecked(word_index)
                    .fetch_and(!(A::Value::ONE << bit_index), ordering);
            }
        }
    }

    /// Sets the bit of given index to the given value and returns the previous
    /// value, without bound checks.
    ///
    /// This method performs a single atomic operation with the given memory
    /// ordering.
    ///
    /// # Safety
    ///
    /// `index` must be between 0 (included) and [`BitLength::len`] (excluded),
    /// and `self.as_ref()` must contain the atomic word holding `index`.
    #[inline]
    unsafe fn swap_unchecked(&self, index: usize, value: bool, ordering: Ordering) -> bool {
        let bits_per_word = A::Value::BITS as usize;
        let word_index = index / bits_per_word;
        let bit_index = index % bits_per_word;
        let bits = self.as_ref();

        let old_word = unsafe {
            if value {
                bits.get_unchecked(word_index)
                    .fetch_or(A::Value::ONE << bit_index, ordering)
            } else {
                bits.get_unchecked(word_index)
                    .fetch_and(!(A::Value::ONE << bit_index), ordering)
            }
        };

        (old_word >> bit_index) & A::Value::ONE != A::Value::ZERO
    }

    /// Sets all bits to the given value.
    fn fill(&mut self, value: bool, ordering: Ordering) {
        let bits_per_word = A::Value::BITS as usize;
        let full_words = self.len() / bits_per_word;
        let residual = self.len() % bits_per_word;
        let bits = self.as_ref();
        let word_value: A::Value = if value {
            !A::Value::ZERO
        } else {
            A::Value::ZERO
        };
        // Just to be sure, add a fence to ensure that we will see all the final
        // values
        core::sync::atomic::fence(Ordering::SeqCst);
        bits[..full_words]
            .iter()
            .for_each(|x| x.store(word_value, ordering));
        if residual != 0 {
            // Use RMW operations: a plain load+store pair would lose a
            // concurrent update to the residual word and reject
            // store-invalid orderings such as Release for the load.
            let mask = (A::Value::ONE << residual) - A::Value::ONE;
            if value {
                bits[full_words].fetch_or(mask, ordering);
            } else {
                bits[full_words].fetch_and(!mask, ordering);
            }
        }
    }

    /// Sets all bits to the given value using a parallel implementation.
    #[cfg(feature = "rayon")]
    fn par_fill(&mut self, value: bool, ordering: Ordering) {
        let bits_per_word = A::Value::BITS as usize;
        let full_words = self.len() / bits_per_word;
        let residual = self.len() % bits_per_word;
        let bits = self.as_ref();
        let word_value: A::Value = if value {
            !A::Value::ZERO
        } else {
            A::Value::ZERO
        };

        // Just to be sure, add a fence to ensure that we will see all the final
        // values
        core::sync::atomic::fence(Ordering::SeqCst);
        bits[..full_words]
            .par_iter()
            .with_len(crate::RAYON_MIN_LEN)
            .for_each(|x| x.store(word_value, ordering));
        if residual != 0 {
            // See fill: RMW keeps the residual update atomic and
            // ordering-agnostic.
            let mask = (A::Value::ONE << residual) - A::Value::ONE;
            if value {
                bits[full_words].fetch_or(mask, ordering);
            } else {
                bits[full_words].fetch_and(!mask, ordering);
            }
        }
    }

    /// Sets all bits to zero.
    fn reset(&mut self, ordering: Ordering) {
        self.fill(false, ordering);
    }

    /// Sets all bits to zero using a parallel implementation.
    #[cfg(feature = "rayon")]
    fn par_reset(&mut self, ordering: Ordering) {
        self.par_fill(false, ordering);
    }

    /// Flips all bits.
    fn flip(&mut self, ordering: Ordering) {
        let bits_per_word = A::Value::BITS as usize;
        let full_words = self.len() / bits_per_word;
        let residual = self.len() % bits_per_word;
        let bits = self.as_ref();
        // Just to be sure, add a fence to ensure that we will see all the final
        // values
        core::sync::atomic::fence(Ordering::SeqCst);
        bits[..full_words]
            .iter()
            .for_each(|x| _ = x.fetch_xor(!A::Value::ZERO, ordering));
        if residual != 0 {
            // Use fetch_xor like the full words: a load+store pair would lose
            // a concurrent update and reject orderings such as Release.
            let mask = (A::Value::ONE << residual) - A::Value::ONE;
            bits[full_words].fetch_xor(mask, ordering);
        }
    }

    /// Flips all bits using a parallel implementation.
    #[cfg(feature = "rayon")]
    fn par_flip(&mut self, ordering: Ordering) {
        let bits_per_word = A::Value::BITS as usize;
        let full_words = self.len() / bits_per_word;
        let residual = self.len() % bits_per_word;
        let bits = self.as_ref();
        // Just to be sure, add a fence to ensure that we will see all the final
        // values
        core::sync::atomic::fence(Ordering::SeqCst);
        bits[..full_words]
            .par_iter()
            .with_len(crate::RAYON_MIN_LEN)
            .for_each(|x| _ = x.fetch_xor(!A::Value::ZERO, ordering));
        if residual != 0 {
            // See flip: RMW keeps the residual update atomic and
            // ordering-agnostic.
            let mask = (A::Value::ONE << residual) - A::Value::ONE;
            bits[full_words].fetch_xor(mask, ordering);
        }
    }

    /// Returns the number of ones in the bit vector, counted in parallel.
    ///
    /// Dirty bits past the logical length are masked out.
    #[cfg(feature = "rayon")]
    fn par_count_ones(&self) -> usize {
        let bits_per_word = A::Value::BITS as usize;
        let full_words = self.len() / bits_per_word;
        let residual = self.len() % bits_per_word;
        let bits = self.as_ref();
        let mut num_ones;
        // Just to be sure, add a fence to ensure that we will see all the final
        // values
        core::sync::atomic::fence(Ordering::SeqCst);
        num_ones = bits[..full_words]
            .par_iter()
            .with_len(crate::RAYON_MIN_LEN)
            .map(|x| x.load(Ordering::Relaxed).count_ones() as usize)
            .sum();
        if residual != 0 {
            num_ones += (bits[full_words].load(Ordering::Relaxed) << (bits_per_word - residual))
                .count_ones() as usize
        }
        num_ones
    }

    /// Returns an iterator over the bits of this atomic bit vector.
    ///
    /// Note that modifying the bit vector while iterating over it will lead to
    /// behavior depending on processor scheduling and memory model.
    /// Nonetheless, all returned values have been valid at some point during
    /// the iteration.
    #[inline(always)]
    fn iter(&self) -> AtomicBitIter<'_, A> {
        AtomicBitIter::new(self.as_ref(), self.len())
    }
}

/// An iterator over the bits of an atomic bit vector as booleans.
///
/// Note that modifying the bit vector while iterating over it will lead to
/// behavior depending on processor scheduling and memory model.
#[derive(Debug, MemSize, MemDbg)]
pub struct AtomicBitIter<'a, A> {
    bits: &'a [A],
    len: usize,
    next_bit_pos: usize,
}

impl<'a, A: PrimitiveAtomicUnsigned<Value: Word>> AtomicBitIter<'a, A> {
    /// Creates an iterator over the first `len` atomic bits in `bits`.
    ///
    /// The backing slice is snapshotted once, so iteration is sound even if the
    /// `AsRef` implementation does not return the same slice on every call.
    ///
    /// # Panics
    ///
    /// Panics if the backing slice cannot hold `len` bits.
    pub fn new<B: ?Sized + AsRef<[A]>>(bits: &'a B, len: usize) -> Self {
        let bits = bits.as_ref();
        let bits_per_word = usize::try_from(A::Value::BITS).expect("word width fits in usize");
        assert!(
            len.div_ceil(bits_per_word) <= bits.len(),
            "Atomic bit-vector backing storage is too short for {len} bits"
        );
        AtomicBitIter {
            bits,
            len,
            next_bit_pos: 0,
        }
    }
}

impl<A: PrimitiveAtomicUnsigned<Value: Word>> Iterator for AtomicBitIter<'_, A> {
    type Item = bool;
    fn next(&mut self) -> Option<bool> {
        if self.next_bit_pos == self.len {
            return None;
        }
        let bits_per_word = usize::try_from(A::Value::BITS).expect("word width fits in usize");
        let word_idx = self.next_bit_pos / bits_per_word;
        let bit_idx = self.next_bit_pos % bits_per_word;
        // SAFETY: `new` checked `len.div_ceil(bits_per_word) <= self.bits.len()`
        // against this exact snapshotted slice, and `next_bit_pos < len` here, so
        // `word_idx < self.bits.len()`.
        let word = unsafe { self.bits.get_unchecked(word_idx).load(Ordering::Relaxed) };
        let bit = (word >> bit_idx) & A::Value::ONE;
        self.next_bit_pos += 1;
        Some(bit != A::Value::ZERO)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let rem = self.len - self.next_bit_pos;
        (rem, Some(rem))
    }
}

impl<A: PrimitiveAtomicUnsigned<Value: Word>> ExactSizeIterator for AtomicBitIter<'_, A> {
    fn len(&self) -> usize {
        self.len - self.next_bit_pos
    }
}

impl<A: PrimitiveAtomicUnsigned<Value: Word>> FusedIterator for AtomicBitIter<'_, A> {}
