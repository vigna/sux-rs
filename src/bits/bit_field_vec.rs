/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Vectors of bit fields of fixed width (AKA "compact arrays", "bit array",
//! etc.)
//!
//! Elements are stored contiguously, with no padding bits (in particular,
//! unless the bit width is a power of two some elements will be stored across
//! word boundaries).
//!
//! There are two flavors: [`BitFieldVec`], a mutable bit-field vector, and
//! [`AtomicBitFieldVec`], a mutable, thread-safe bit-field vector.
//!
//! These flavors depend on a backend, and presently we provide, given an
//! unsigned integer type `W` or an unsigned atomic integer type `A`:
//!
//! - `BitFieldVec<Vec<T>>`: a mutable, growable and resizable bit-field vector;
//! - `BitFieldVec<AsRef<[W]>>`: an immutable bit-field vector, useful for
//!   [ε-serde](https://crates.io/crates/epserde) support;
//! - `BitFieldVec<AsRef<[W]> + AsMut<[W]>>`: a mutable (but not resizable) bit
//!   vector;
//! - `AtomicBitFieldVec<AsRef<[A]>>`: a partially thread-safe, mutable (but not
//!   resizable) bit-field vector.
//!
//! More generally, the underlying type must satisfy the trait [`Word`] for
//! [`BitFieldVec`], while for [`AtomicBitFieldVec`] it must satisfy
//! [`PrimitiveAtomic`] with a
//! [`Value`](atomic_primitive::PrimitiveAtomic::Value) satisfying [`Word`].
//! A blanket implementation exposes slices of elements of type `W` as bit-field
//! vectors of width `W::BITS`, analogously for atomic types `A`.
//!
//! The traits [`BitFieldSlice`] and [`BitFieldSliceMut`] provide a uniform
//! interface to access to the content of (a reference to) the bit-field vector.
//! There is also a [`AtomicBitFieldSlice`] trait for atomic bit-field vectors.
//! Since they are also implemented for slices of words, they make it easy to
//! write generic code that works both on bit-field vectors and on slices of
//! words when you need to consider the bit width of each element.
//!
//! Note that the [`try_chunks_mut`](SliceByValueMut::try_chunks_mut) method is
//! part of the [`SliceByValueMut`] trait, and thus returns an iterator over
//! elements implementing [`SliceByValueMut`]; the elements, however, implement
//! also [`BitFieldSliceMut`], and you can use this property by adding the bound
//! `for<'a> BitFieldSliceMut<ChunksMut<'a>: Iterator<Item:
//! BitFieldSliceMut>>`.
//!
//! Nothing is assumed about the content of the backend outside the
//! bits of the vector. Moreover, the content of the backend outside of the
//! vector is never modified by the methods of this structure.
//!
//! For high-speed unchecked scanning, we implement [`IntoUncheckedIterator`]
//! and [`IntoUncheckedBackIterator`] on a reference to this type. They are
//! used, for example, to provide
//! [predecessor](crate::traits::indexed_dict::Pred) and
//! [successor](crate::traits::indexed_dict::Succ) primitives for
//! [`EliasFano`].
//!
//! # Low-level support
//!
//! The methods [`addr_of`](BitFieldVec::addr_of) and
//! [`get_unaligned`](BitFieldVec::get_unaligned) can be used to manually
//! prefetch parts of the data structure, or read values using unaligned read,
//! when the bit width makes it possible.
//!
//! The wrapper [`BitFieldVecU`] implements [`SliceByValue`] using
//! unaligned reads and delegates all iterator methods. It can be just plugged
//! in place of a normal [`BitFieldVec`] when the trait bound is
//! [`SliceByValue`].
//!
//! # Examples
//!
//! ```
//! # use sux::prelude::*;
//! # use bit_field_slice::*;
//! # use value_traits::slices::*;
//! // Bit field vector of bit width 5 and length 10, all entries set to zero
//! let mut b = <BitFieldVec<Vec<usize>>>::new(5, 10);
//! assert_eq!(b.len(), 10);
//! assert_eq!(b.bit_width(), 5);
//! b.set_value(0, 3);
//! assert_eq!(b.index_value(0), 3);
//!
//! // Empty bit field vector of bit width 20 with capacity 10
//! let mut b = <BitFieldVec<Vec<usize>>>::with_capacity(20, 10);
//! assert_eq!(b.len(), 0);
//! assert_eq!(b.bit_width(), 20);
//! b.push(20);
//! assert_eq!(b.len(), 1);
//! assert_eq!(b.index_value(0), 20);
//! assert_eq!(b.pop(), Some(20));
//!
//! // Convenience macro
//! let b = bit_field_vec![10; 4, 500, 2, 0, 1];
//! assert_eq!(b.len(), 5);
//! assert_eq!(b.bit_width(), 10);
//! assert_eq!(b.index_value(0), 4);
//! assert_eq!(b.index_value(1), 500);
//! assert_eq!(b.index_value(2), 2);
//! assert_eq!(b.index_value(3), 0);
//! assert_eq!(b.index_value(4), 1);
//! ```

use crate::prelude::{bit_field_slice::*, *};
use crate::traits::ambassador_impl_Backend;
use crate::traits::{Backend, Word};
use crate::utils::PrimitiveUnsignedExt;
use crate::utils::{
    CannotCastToAtomicError, transmute_boxed_slice_from_atomic, transmute_boxed_slice_into_atomic,
    transmute_vec_from_atomic, transmute_vec_into_atomic,
};
use crate::{panic_if_out_of_bounds, panic_if_value};
use ambassador::Delegate;
use anyhow::{Result, bail};
use atomic_primitive::{Atomic, AtomicPrimitive, PrimitiveAtomic, PrimitiveAtomicUnsigned};
use mem_dbg::*;
use num_primitive::{PrimitiveInteger, PrimitiveNumber, PrimitiveNumberAs};
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use std::iter::FusedIterator;
use std::sync::atomic::{Ordering, compiler_fence, fence};
use value_traits::slices::{SliceByValue, SliceByValueMut};

/// Convenient, [`vec!`]-like macro to initialize [`usize`]-based
/// bit-field vectors.
///
/// - `bit_field_vec![width]`: creates an empty bit-field vector of given bit
///   width.
///
/// - `bit_field_vec![width => value; length]`: creates a bit-field vector of
///   given bit width and length, with all entries set to `value`.
///
/// - `bit_field_vec![width; v₀, v₁, … ]`: creates a bit-field vector of
///   given bit width with entries set to `v₀`, `v₁`, ….
///
/// # Examples
///
/// ```
/// # use sux::prelude::*;
/// # use bit_field_slice::*;
/// # use value_traits::slices::*;
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
/// assert_eq!(b.index_value(0), 4);
/// assert_eq!(b.index_value(1), 500);
/// assert_eq!(b.index_value(2), 2);
/// assert_eq!(b.index_value(3), 0);
/// assert_eq!(b.index_value(4), 1);
/// ```
#[macro_export]
macro_rules! bit_field_vec {
    ($w:expr) => {
        <$crate::bits::BitFieldVec>::new($w, 0)
    };
    ($w:expr; $n:expr; $v:expr) => {
        compile_error!(
            "the syntax bit_field_vec![width; length; value] has been removed: use bit_field_vec![width => value; length] instead"
        )
    };
    ($w:expr => $v:expr; $n:expr) => {
        {
            let mut bit_field_vec = <$crate::bits::BitFieldVec>::with_capacity($w, $n);
            // Force type
            let v: usize = $v;
            bit_field_vec.resize($n, v);
            bit_field_vec
        }
    };
    ($w:expr; $($x:expr),+ $(,)?) => {
        {
            let mut b = <$crate::bits::BitFieldVec>::with_capacity($w, [$($x),+].len());
            $(
                // Force type
                let x: usize = $x;
                b.push(x);
            )*
            b
        }
    };
}

/// A vector of bit fields of fixed width (AKA "compact array", "bit array",
/// etc.).
///
/// See the [module documentation](crate::bits) for more details.
#[derive(Debug, Clone, Hash, MemSize, MemDbg, Delegate, value_traits::Subslices)]
#[value_traits_subslices(bound = "B: AsRef<[B::Word]>")]
#[value_traits_subslices(bound = "B::Word: Word")]
#[derive(value_traits::SubslicesMut)]
#[value_traits_subslices_mut(bound = "B: AsRef<[B::Word]> + AsMut<[B::Word]>")]
#[value_traits_subslices_mut(bound = "B::Word: Word")]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde))]
#[cfg_attr(
    feature = "epserde",
    epserde(bound(
        deser = "for<'a> <B as epserde::deser::DeserInner>::DeserType<'a>: Backend<Word = B::Word>"
    ))
)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[delegate(crate::traits::Backend, target = "bits")]
pub struct BitFieldVec<B: Backend = Vec<usize>> {
    /// The underlying storage.
    bits: B,
    /// The bit width of the values stored in the vector.
    bit_width: usize,
    /// A mask with its lowest `bit_width` bits set to one.
    mask: B::Word,
    /// The length of the vector.
    len: usize,
}

/// Robust, heavily checked mask function for constructors and similar methods.
fn mask<W: Word>(bit_width: usize) -> W {
    if bit_width == 0 {
        W::ZERO
    } else {
        W::MAX
            >> (W::BITS as usize)
                .checked_sub(bit_width)
                .expect("bit_width > W::BITS as usize")
    }
}

impl<B: Backend<Word: Word>> BitFieldVec<B> {
    /// # Safety
    /// `len` * `bit_width` must be between 0 (included) and the number of
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

    /// Returns the backend, the bit width, and the length, consuming this
    /// vector.
    #[inline(always)]
    pub fn into_raw_parts(self) -> (B, usize, usize) {
        (self.bits, self.bit_width, self.len)
    }

    /// Returns the mask used to extract values from the vector.
    /// This will keep the lowest `bit_width` bits.
    pub const fn mask(&self) -> B::Word {
        self.mask
    }

    /// Replaces the backend by applying a function, consuming this vector.
    ///
    /// # Safety
    /// The caller must ensure that the length is compatible with the new
    /// backend.
    #[inline(always)]
    pub unsafe fn map<B2: Backend<Word: Word>>(self, f: impl FnOnce(B) -> B2) -> BitFieldVec<B2> {
        BitFieldVec {
            bits: f(self.bits),
            bit_width: self.bit_width,
            mask: mask(self.bit_width),
            len: self.len,
        }
    }
}

impl<B: Backend<Word: Word> + AsRef<[B::Word]>> BitFieldVec<B> {
    /// Gets the address of the item storing (the first part of)
    /// the element of given index.
    ///
    /// This method is mainly useful for manually prefetching
    /// parts of the data structure.
    pub fn addr_of(&self, index: usize) -> *const B::Word {
        let start_bit = index * self.bit_width;
        let word_index = start_bit / B::Word::BITS as usize;
        (&self.bits.as_ref()[word_index]) as *const _
    }

    /// Like [`SliceByValue::index_value`], but using unaligned reads.
    ///
    /// This method can be used only for bit width smaller than or equal to
    /// `W::BITS as usize - 8 + 2` or equal to `W::BITS as usize - 8 + 4` or `W::BITS`. Moreover,
    /// an additional padding word must be present at the end of the vector.
    ///
    /// Note that to guarantee the absence of undefined behavior this method
    /// has to perform several tests. Consider using
    /// [`get_unaligned_unchecked`](Self::get_unaligned_unchecked) if you are
    /// sure that the constraints are respected.
    ///
    /// # Panics
    ///
    /// This method will panic if the constraints above are not respected.
    pub fn get_unaligned(&self, index: usize) -> B::Word {
        assert_unaligned!(B::Word, self.bit_width);
        panic_if_out_of_bounds!(index, self.len);
        // Check that the read_unaligned of size_of::<W>() bytes starting at
        // byte offset start_bit / 8 does not exceed the allocation.
        assert!(
            (index * self.bit_width) / 8 + size_of::<B::Word>()
                <= std::mem::size_of_val(self.bits.as_ref())
        );
        unsafe { self.get_unaligned_unchecked(index) }
    }

    /// Like [`SliceByValue::get_value_unchecked`], but using unaligned reads.
    ///
    /// # Safety
    ///
    /// This method can be used only for bit width smaller than or equal to
    /// `W::BITS as usize - 8 + 2` or equal to `W::BITS as usize - 8 + 4` or `W::BITS`. Moreover,
    /// an additional padding word must be present at the end of the vector,
    /// and `index` must be within bounds.
    ///
    /// # Panics
    ///
    /// This method will panic in debug mode if the safety constraints are not
    /// respected.
    pub unsafe fn get_unaligned_unchecked(&self, index: usize) -> B::Word {
        debug_assert_unaligned!(B::Word, self.bit_width);
        let base_ptr = self.bits.as_ref().as_ptr() as *const u8;
        let start_bit = index * self.bit_width;
        // Check that the read_unaligned of size_of::<W>() bytes starting at
        // byte offset start_bit / 8 does not exceed the allocation.
        debug_assert!(
            start_bit / 8 + size_of::<B::Word>() <= std::mem::size_of_val(self.bits.as_ref())
        );
        let ptr = unsafe { base_ptr.add(start_bit / 8) } as *const B::Word;
        let word = unsafe { core::ptr::read_unaligned(ptr) };
        (word >> (start_bit % 8)) & self.mask
    }

    /// Returns the backend of the vector as a slice of words.
    pub fn as_slice(&self) -> &[B::Word] {
        self.bits.as_ref()
    }
}
/// An iterator over non-overlapping chunks of a bit-field vector, starting at
/// the beginning of the vector.
///
/// When the vector len is not evenly divided by the chunk size, the last chunk
/// of the iteration will be shorter.
///
/// This struct is created by the
/// [`try_chunks_mut`](crate::bits::bit_field_vec::BitFieldVec#impl-BitFieldSliceMut-for-BitFieldVec<B>)
/// method.
pub struct ChunksMut<'a, W: Word> {
    remaining: usize,
    bit_width: usize,
    chunk_size: usize,
    iter: std::slice::ChunksMut<'a, W>,
}

impl<'a, W: Word> Iterator for ChunksMut<'a, W> {
    type Item = BitFieldVec<&'a mut [W]>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|chunk| {
            let size = Ord::min(self.chunk_size, self.remaining);
            let next = unsafe { BitFieldVec::from_raw_parts(chunk, self.bit_width, size) };
            self.remaining -= size;
            next
        })
    }
}

impl<'a, W: Word> ExactSizeIterator for ChunksMut<'a, W> where
    std::slice::ChunksMut<'a, W>: ExactSizeIterator
{
}

impl<'a, W: Word> FusedIterator for ChunksMut<'a, W> where
    std::slice::ChunksMut<'a, W>: FusedIterator
{
}

impl<B: Backend<Word: Word> + AsRef<[B::Word]>> BitFieldVec<B> {}

impl<W: Word> BitFieldVec<Vec<W>> {
    /// Creates a new zero-initialized vector of given bit width and length.
    pub fn new(bit_width: usize, len: usize) -> Self {
        // We need at least one word to handle the case of bit width zero.
        let n_of_words = Ord::max(1, (len * bit_width).div_ceil(W::BITS as usize));
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
        let n_of_words = Ord::max(1, (capacity * bit_width).div_ceil(W::BITS as usize));
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
    /// `len * bit_width` must be at most `self.bits.len() * W::BITS as usize`. Note that
    /// setting the length might result in reading uninitialized data.
    pub unsafe fn set_len(&mut self, len: usize) {
        debug_assert!(len * self.bit_width <= self.bits.len() * W::BITS as usize);
        self.len = len;
    }

    /// Sets len to 0.
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Returns the bit width of the values inside the vector.
    pub const fn bit_width(&self) -> usize {
        debug_assert!(self.bit_width <= W::BITS as usize);
        self.bit_width
    }

    /// Creates a new vector by copying a slice; the bit width will be the minimum
    /// width sufficient to hold all values in the slice.
    ///
    /// Returns an error if the bit width of the values in `slice` is larger than
    /// `W::BITS`.
    pub fn from_slice<S: BitFieldSlice<Value: Word + PrimitiveNumberAs<W>>>(
        slice: &S,
    ) -> Result<Self> {
        let mut max_len: usize = 0;
        for i in 0..slice.len() {
            max_len = Ord::max(max_len, unsafe {
                slice.get_value_unchecked(i).bit_len() as usize
            });
        }

        if max_len > W::BITS as usize {
            bail!(
                "Cannot convert a slice of bit width {} into a slice with W = {}",
                max_len,
                std::any::type_name::<W>()
            );
        }
        let mut result = Self::new(max_len, slice.len());
        for i in 0..slice.len() {
            unsafe { result.set_value_unchecked(i, slice.get_value_unchecked(i).as_to::<W>()) };
        }

        Ok(result)
    }

    /// Adds a value at the end of the vector.
    pub fn push(&mut self, value: W) {
        panic_if_value!(value, self.mask, self.bit_width);
        if (self.len + 1) * self.bit_width > self.bits.len() * W::BITS as usize {
            self.bits.push(W::ZERO);
        }
        unsafe {
            self.set_value_unchecked(self.len, value);
        }
        self.len += 1;
    }

    /// Truncates or extends with `value` the vector.
    pub fn resize(&mut self, new_len: usize, value: W) {
        panic_if_value!(value, self.mask, self.bit_width);
        if new_len > self.len {
            if new_len * self.bit_width > self.bits.len() * W::BITS as usize {
                self.bits.resize(
                    (new_len * self.bit_width).div_ceil(W::BITS as usize),
                    W::ZERO,
                );
            }
            for i in self.len..new_len {
                unsafe {
                    self.set_value_unchecked(i, value);
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
        let value = self.index_value(self.len - 1);
        self.len -= 1;
        Some(value)
    }

    /// Ensures a padding word is present at the end and converts the
    /// backend to `Box<[W]>`.
    ///
    /// The extra word ensures that unaligned reads of `size_of::<W>()`
    /// bytes starting at any byte offset within the data never exceed the
    /// allocation. If the allocation already has more words than needed
    /// for the data, no word is added.
    pub fn into_padded(mut self) -> BitFieldVec<Box<[W]>> {
        let needed = (self.len * self.bit_width).div_ceil(W::BITS as usize);
        if self.bits.len() <= needed {
            self.bits.push(W::ZERO);
        }
        unsafe {
            BitFieldVec::from_raw_parts(self.bits.into_boxed_slice(), self.bit_width, self.len)
        }
    }
}

impl<W: Word> BitFieldVec<Box<[W]>> {
    /// Creates a new zero-initialized vector of given bit width and length,
    /// with a padding word at the end for safe unaligned reads.
    ///
    /// This constructor is useful for structures implementing
    /// [`TryIntoUnaligned`](crate::traits::TryIntoUnaligned) that want to avoid
    /// reallocations.
    pub fn new_padded(bit_width: usize, len: usize) -> Self {
        let n_of_words = (len * bit_width).div_ceil(W::BITS as usize);
        Self {
            bits: vec![W::ZERO; n_of_words + 1].into_boxed_slice(),
            bit_width,
            mask: mask(bit_width),
            len,
        }
    }
}

impl<B: Backend<Word: Word> + AsRef<[B::Word]> + AsMut<[B::Word]>> BitFieldVec<B> {}

impl<B: Backend<Word: Word>> BitWidth for BitFieldVec<B> {
    #[inline(always)]
    fn bit_width(&self) -> usize {
        debug_assert!(self.bit_width <= B::Word::BITS as usize);
        self.bit_width
    }
}

impl<B: Backend<Word: Word> + AsRef<[B::Word]>> SliceByValue for BitFieldVec<B> {
    type Value = B::Word;
    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }

    unsafe fn get_value_unchecked(&self, index: usize) -> B::Word {
        let bits = B::Word::BITS as usize;
        let pos = index * self.bit_width;
        let word_index = pos / bits;
        let bit_index = pos % bits;
        let data = self.bits.as_ref();

        unsafe {
            if bit_index + self.bit_width <= bits {
                (*data.get_unchecked(word_index) >> bit_index) & self.mask
            } else {
                ((*data.get_unchecked(word_index) >> bit_index)
                    | (*data.get_unchecked(word_index + 1) << (bits - bit_index)))
                    & self.mask
            }
        }
    }
}

impl<B: Backend<Word: Word> + AsRef<[B::Word]>> BitFieldSlice for BitFieldVec<B> {
    fn as_slice(&self) -> &[Self::Value] {
        self.bits.as_ref()
    }
}

impl<B: Backend<Word: Word> + AsRef<[B::Word]> + AsMut<[B::Word]>> BitFieldSliceMut
    for BitFieldVec<B>
{
    fn reset(&mut self) {
        let bit_len = self.len * self.bit_width;
        let full_words = bit_len / B::Word::BITS as usize;
        let residual = bit_len % B::Word::BITS as usize;
        let bits = self.bits.as_mut();
        bits[..full_words]
            .iter_mut()
            .for_each(|x| *x = B::Word::ZERO);
        if residual != 0 {
            bits[full_words] &= B::Word::MAX << residual;
        }
    }

    #[cfg(feature = "rayon")]
    fn par_reset(&mut self) {
        let bit_len = self.len * self.bit_width;
        let full_words = bit_len / B::Word::BITS as usize;
        let residual = bit_len % B::Word::BITS as usize;
        let bits = self.bits.as_mut();
        bits[..full_words]
            .par_iter_mut()
            .with_min_len(crate::RAYON_MIN_LEN)
            .for_each(|x| *x = B::Word::ZERO);
        if residual != 0 {
            bits[full_words] &= B::Word::MAX << residual;
        }
    }

    fn as_mut_slice(&mut self) -> &mut [B::Word] {
        self.bits.as_mut()
    }
}

/// Error type returned when [`try_chunks_mut`](SliceByValueMut::try_chunks_mut)
/// does not find sufficient alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunksMutError<W: Word> {
    bit_width: usize,
    chunk_size: usize,
    _marker: core::marker::PhantomData<W>,
}

impl<W: Word> core::fmt::Display for ChunksMutError<W> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "try_chunks_mut needs that the bit width ({}) times the chunk size ({}) is a multiple of W::BITS ({}) to return more than one chunk",
            self.bit_width,
            self.chunk_size,
            W::BITS as usize
        )
    }
}

impl<W: Word> std::error::Error for ChunksMutError<W> {}

impl<B: Backend<Word: Word> + AsRef<[B::Word]> + AsMut<[B::Word]>> SliceByValueMut
    for BitFieldVec<B>
{
    #[inline(always)]
    fn set_value(&mut self, index: usize, value: B::Word) {
        panic_if_out_of_bounds!(index, self.len);
        panic_if_value!(value, self.mask, self.bit_width);
        unsafe {
            self.set_value_unchecked(index, value);
        }
    }

    unsafe fn set_value_unchecked(&mut self, index: usize, value: B::Word) {
        let bits = B::Word::BITS as usize;
        let pos = index * self.bit_width;
        let word_index = pos / bits;
        let bit_index = pos % bits;
        let data = self.bits.as_mut();

        unsafe {
            if bit_index + self.bit_width <= bits {
                let mut word = *data.get_unchecked_mut(word_index);
                word &= !(self.mask << bit_index);
                word |= value << bit_index;
                *data.get_unchecked_mut(word_index) = word;
            } else {
                let mut word = *data.get_unchecked_mut(word_index);
                word &= (B::Word::ONE << bit_index) - B::Word::ONE;
                word |= value << bit_index;
                *data.get_unchecked_mut(word_index) = word;

                let mut word = *data.get_unchecked_mut(word_index + 1);
                word &= !(self.mask >> (bits - bit_index));
                word |= value >> (bits - bit_index);
                *data.get_unchecked_mut(word_index + 1) = word;
            }
        }
    }

    fn replace_value(&mut self, index: usize, value: B::Word) -> B::Word {
        panic_if_out_of_bounds!(index, self.len);
        panic_if_value!(value, self.mask, self.bit_width);
        unsafe { self.replace_value_unchecked(index, value) }
    }

    unsafe fn replace_value_unchecked(&mut self, index: usize, value: B::Word) -> B::Word {
        let bits = B::Word::BITS as usize;
        let pos = index * self.bit_width;
        let word_index = pos / bits;
        let bit_index = pos % bits;
        let data = self.bits.as_mut();

        unsafe {
            if bit_index + self.bit_width <= bits {
                let mut word = *data.get_unchecked_mut(word_index);
                let old_value = (word >> bit_index) & self.mask;
                word &= !(self.mask << bit_index);
                word |= value << bit_index;
                *data.get_unchecked_mut(word_index) = word;
                old_value
            } else {
                let mut word = *data.get_unchecked_mut(word_index);
                let mut old_value = word >> bit_index;
                word &= (B::Word::ONE << bit_index) - B::Word::ONE;
                word |= value << bit_index;
                *data.get_unchecked_mut(word_index) = word;

                let mut word = *data.get_unchecked_mut(word_index + 1);
                old_value |= word << (bits - bit_index);
                word &= !(self.mask >> (bits - bit_index));
                word |= value >> (bits - bit_index);
                *data.get_unchecked_mut(word_index + 1) = word;
                old_value & self.mask
            }
        }
    }

    /// This implementation performs the copy word by word, which is
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
        let bits = B::Word::BITS as usize;
        let src_bit = src_pos % bits;
        let dst_bit = dst_pos % bits;
        let src_first_word = src_pos / bits;
        let dst_first_word = dst_pos / bits;
        let src_last_word = (src_pos + bit_len - 1) / bits;
        let dst_last_word = (dst_pos + bit_len - 1) / bits;
        let source = self.bits.as_ref();
        let dest = dst.bits.as_mut();

        if src_first_word == src_last_word && dst_first_word == dst_last_word {
            let mask = B::Word::MAX >> (bits - bit_len);
            let word = (source[src_first_word] >> src_bit) & mask;
            dest[dst_first_word] &= !(mask << dst_bit);
            dest[dst_first_word] |= word << dst_bit;
        } else if src_first_word == src_last_word {
            // dst_first_word != dst_last_word
            let mask = B::Word::MAX >> (bits - bit_len);
            let word = (source[src_first_word] >> src_bit) & mask;
            dest[dst_first_word] &= !(mask << dst_bit);
            dest[dst_first_word] |= (word & mask) << dst_bit;
            dest[dst_last_word] &= !(mask >> (bits - dst_bit));
            dest[dst_last_word] |= (word & mask) >> (bits - dst_bit);
        } else if dst_first_word == dst_last_word {
            // src_first_word != src_last_word
            let mask = B::Word::MAX >> (bits - bit_len);
            let word = ((source[src_first_word] >> src_bit)
                | (source[src_last_word] << (bits - src_bit)))
                & mask;
            dest[dst_first_word] &= !(mask << dst_bit);
            dest[dst_first_word] |= word << dst_bit;
        } else if src_bit == dst_bit {
            // src_first_word != src_last_word && dst_first_word != dst_last_word
            let mask = B::Word::MAX << dst_bit;
            dest[dst_first_word] &= !mask;
            dest[dst_first_word] |= source[src_first_word] & mask;

            dest[(1 + dst_first_word)..dst_last_word]
                .copy_from_slice(&source[(1 + src_first_word)..src_last_word]);

            let residual = bit_len - (bits - src_bit) - (dst_last_word - dst_first_word - 1) * bits;
            let mask = B::Word::MAX >> (bits - residual);
            dest[dst_last_word] &= !mask;
            dest[dst_last_word] |= source[src_last_word] & mask;
        } else if src_bit < dst_bit {
            // src_first_word != src_last_word && dst_first_word !=
            // dst_last_word
            let dst_mask = B::Word::MAX << dst_bit;
            let src_mask = B::Word::MAX << src_bit;
            let shift = dst_bit - src_bit;
            dest[dst_first_word] &= !dst_mask;
            dest[dst_first_word] |= (source[src_first_word] & src_mask) << shift;

            let mut word = source[src_first_word] >> (bits - shift);
            for i in 1..dst_last_word - dst_first_word {
                dest[dst_first_word + i] = word | (source[src_first_word + i] << shift);
                word = source[src_first_word + i] >> (bits - shift);
            }
            let residual = bit_len - (bits - dst_bit) - (dst_last_word - dst_first_word - 1) * bits;
            let mask = B::Word::MAX >> (bits - residual);
            dest[dst_last_word] &= !mask;
            dest[dst_last_word] |= (word | (source[src_last_word] << shift)) & mask;
        } else {
            // src_first_word != src_last_word && dst_first_word !=
            // dst_last_word && src_bit > dst_bit
            let dst_mask = B::Word::MAX << dst_bit;
            let src_mask = B::Word::MAX << src_bit;
            let shift = src_bit - dst_bit;
            dest[dst_first_word] &= !dst_mask;
            dest[dst_first_word] |= (source[src_first_word] & src_mask) >> shift;
            dest[dst_first_word] |= source[src_first_word + 1] << (bits - shift);

            let mut word = source[src_first_word + 1] >> shift;

            for i in 1..dst_last_word - dst_first_word {
                word |= source[src_first_word + i + 1] << (bits - shift);
                dest[dst_first_word + i] = word;
                word = source[src_first_word + i + 1] >> shift;
            }

            word |= source[src_last_word] << (bits - shift);

            let residual = bit_len - (bits - dst_bit) - (dst_last_word - dst_first_word - 1) * bits;
            let mask = B::Word::MAX >> (bits - residual);
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
        F: FnMut(Self::Value) -> Self::Value,
    {
        if self.is_empty() {
            return;
        }
        let bit_width = self.bit_width();
        if bit_width == 0 {
            return;
        }
        let mask = self.mask;
        let number_of_words: usize = self.bits.as_ref().len();
        let last_word_idx = number_of_words.saturating_sub(1);

        let bits = B::Word::BITS as usize;
        let mut write_buffer: B::Word = B::Word::ZERO;
        let mut read_buffer: B::Word = *unsafe { self.bits.as_ref().get_unchecked(0) };

        // specialized case because it's much faster
        if bit_width.is_power_of_two() {
            let mut bits_in_buffer = 0;

            let mut buffer_limit = (self.len() * bit_width) % bits;
            if buffer_limit == 0 {
                buffer_limit = bits;
            }

            for read_idx in 1..number_of_words {
                // pre-load the next word so it loads while we parse the buffer
                let next_word: B::Word = *unsafe { self.bits.as_ref().get_unchecked(read_idx) };

                // parse as much as we can from the buffer
                loop {
                    let next_bits_in_buffer = bits_in_buffer + bit_width;

                    if next_bits_in_buffer > bits {
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

                debug_assert_eq!(read_buffer, B::Word::ZERO);
                *unsafe { self.bits.as_mut().get_unchecked_mut(read_idx - 1) } = write_buffer;
                read_buffer = next_word;
                write_buffer = B::Word::ZERO;
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
                bits_in_buffer += bit_width;
            }

            *unsafe { self.bits.as_mut().get_unchecked_mut(last_word_idx) } = write_buffer;
            return;
        }

        // The position inside the word. In most parametrization of the
        // vector, since the bit_width is not necessarily an integer
        // divisor of the word size, we need to keep track of the position
        // inside the word. As we scroll through the bits, due to the bits
        // remainder, we may need to operate on two words at the same time.
        let mut global_bit_index: usize = 0;

        // The bit-index boundaries of the current word.
        let mut lower_word_limit = 0;
        let mut upper_word_limit = bits;

        // We iterate across the words
        for word_number in 0..last_word_idx {
            // We iterate across the elements in the word.
            while global_bit_index + bit_width <= upper_word_limit {
                // We retrieve the value from the current word.
                let offset = global_bit_index - lower_word_limit;
                global_bit_index += bit_width;
                let element = self.mask & (read_buffer >> offset);

                // We apply the function to the element.
                let new_element = f(element);

                // We set the element in the new word.
                write_buffer |= new_element << offset;
            }

            // We retrieve the next word from the bitvec.
            let next_word = *unsafe { self.bits.as_ref().get_unchecked(word_number + 1) };

            let mut new_write_buffer = B::Word::ZERO;
            if upper_word_limit != global_bit_index {
                let remainder = upper_word_limit - global_bit_index;
                let offset = global_bit_index - lower_word_limit;
                // We compose the element from the remaining elements in the
                // current word and the elements in the next word.
                let element = ((read_buffer >> offset) | (next_word << remainder)) & self.mask;
                global_bit_index += bit_width;

                // We apply the function to the element.
                let new_element = f(element);

                write_buffer |= new_element << offset;

                new_write_buffer = new_element >> remainder;
            };

            read_buffer = next_word;

            *unsafe { self.bits.as_mut().get_unchecked_mut(word_number) } = write_buffer;

            write_buffer = new_write_buffer;
            lower_word_limit = upper_word_limit;
            upper_word_limit += bits;
        }

        let mut offset = global_bit_index - lower_word_limit;

        // We iterate across the elements in the word.
        while offset < self.len() * bit_width - global_bit_index {
            // We retrieve the value from the current word.
            let element = self.mask & (read_buffer >> offset);

            // We apply the function to the element.
            let new_element = f(element);

            // We set the element in the new word.
            write_buffer |= new_element << offset;
            offset += bit_width;
        }

        *unsafe { self.bits.as_mut().get_unchecked_mut(last_word_idx) } = write_buffer;
    }

    type ChunksMut<'a>
        = ChunksMut<'a, B::Word>
    where
        Self: 'a;

    type ChunksMutError = ChunksMutError<B::Word>;

    /// # Errors
    ///
    /// This method will return an error if the chunk size multiplied by
    /// the [bit width](BitWidth::bit_width) is not a multiple of
    /// `W::BITS` and more than one chunk must be returned.
    fn try_chunks_mut(
        &mut self,
        chunk_size: usize,
    ) -> Result<Self::ChunksMut<'_>, ChunksMutError<B::Word>> {
        let len = self.len();
        let bit_width = self.bit_width();
        let bits = B::Word::BITS as usize;
        if len <= chunk_size || (chunk_size * bit_width) % bits == 0 {
            // chunks_mut panics with chunk_size 0, so use 1 when the
            // product is zero (bit_width == 0); the iterator will yield
            // empty slices anyway.
            let words_per_chunk = Ord::max(1, (chunk_size * bit_width).div_ceil(bits));
            Ok(ChunksMut {
                remaining: len,
                bit_width: self.bit_width,
                chunk_size,
                iter: self.bits.as_mut()[..(len * bit_width).div_ceil(bits)]
                    .chunks_mut(words_per_chunk),
            })
        } else {
            Err(ChunksMutError {
                bit_width,
                chunk_size,
                _marker: core::marker::PhantomData,
            })
        }
    }
}

// Support for unchecked iterators

/// An [`UncheckedIterator`] over the values of a [`BitFieldVec`].
#[derive(Debug, Clone, MemSize, MemDbg)]
pub struct BitFieldVecUncheckedIter<'a, B: Backend<Word: Word>> {
    vec: &'a BitFieldVec<B>,
    word_index: usize,
    window: B::Word,
    fill: usize,
}

impl<'a, B: Backend<Word: Word> + AsRef<[B::Word]>> BitFieldVecUncheckedIter<'a, B> {
    fn new(vec: &'a BitFieldVec<B>, index: usize) -> Self {
        if index > vec.len() {
            panic!("Start index out of bounds: {} > {}", index, vec.len());
        }
        let bits = B::Word::BITS as usize;
        let (fill, word_index);
        let window = if index == vec.len() {
            word_index = 0;
            fill = 0;
            B::Word::ZERO
        } else {
            let bit_offset = index * vec.bit_width;
            let bit_index = bit_offset % bits;

            word_index = bit_offset / bits;
            fill = bits - bit_index;
            unsafe {
                // SAFETY: index has been checked at the start and it is within bounds
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

impl<B: Backend<Word: Word> + AsRef<[B::Word]>> crate::traits::UncheckedIterator
    for BitFieldVecUncheckedIter<'_, B>
{
    type Item = B::Word;
    unsafe fn next_unchecked(&mut self) -> B::Word {
        let bit_width = self.vec.bit_width;

        if self.fill >= bit_width {
            self.fill -= bit_width;
            let res = self.window & self.vec.mask;
            self.window >>= bit_width;
            return res;
        }

        let res = self.window;
        self.word_index += 1;
        self.window = *unsafe { self.vec.bits.as_ref().get_unchecked(self.word_index) };
        let res = (res | (self.window << self.fill)) & self.vec.mask;
        let used = bit_width - self.fill;
        self.window >>= used;
        self.fill = B::Word::BITS as usize - used; // not in a loop
        res
    }
}

impl<'a, B: Backend<Word: Word> + AsRef<[B::Word]>> IntoUncheckedIterator for &'a BitFieldVec<B> {
    type Item = B::Word;
    type IntoUncheckedIter = BitFieldVecUncheckedIter<'a, B>;
    fn into_unchecked_iter_from(self, from: usize) -> Self::IntoUncheckedIter {
        BitFieldVecUncheckedIter::new(self, from)
    }
}

/// An [`UncheckedIterator`] moving backwards over the values of a [`BitFieldVec`].
#[derive(Debug, Clone, MemSize, MemDbg)]
pub struct BitFieldVecUncheckedBackIter<'a, B: Backend<Word: Word>> {
    vec: &'a BitFieldVec<B>,
    word_index: usize,
    window: B::Word,
    fill: usize,
}

impl<'a, B: Backend<Word: Word> + AsRef<[B::Word]>> BitFieldVecUncheckedBackIter<'a, B> {
    fn new(vec: &'a BitFieldVec<B>, index: usize) -> Self {
        if index > vec.len() {
            panic!("Start index out of bounds: {} > {}", index, vec.len());
        }
        let bits = B::Word::BITS as usize;
        let (word_index, fill);

        let window = if index == 0 {
            word_index = 0;
            fill = 0;
            B::Word::ZERO
        } else {
            // We have to handle the case of zero bit width
            let bit_offset = (index * vec.bit_width).saturating_sub(1);
            let bit_index = bit_offset % bits;

            word_index = bit_offset / bits;
            fill = bit_index + 1;
            unsafe {
                // SAFETY: index has been checked at the start and it is within bounds
                *vec.bits.as_ref().get_unchecked(word_index) << (bits - fill)
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

impl<B: Backend<Word: Word> + AsRef<[B::Word]>> crate::traits::UncheckedIterator
    for BitFieldVecUncheckedBackIter<'_, B>
{
    type Item = B::Word;
    unsafe fn next_unchecked(&mut self) -> B::Word {
        let bit_width = self.vec.bit_width;

        if self.fill >= bit_width {
            self.fill -= bit_width;
            self.window = self.window.rotate_left(bit_width as u32);
            return self.window & self.vec.mask;
        }

        let mut res = self.window.rotate_left(self.fill as u32);
        self.word_index -= 1;
        self.window = *unsafe { self.vec.bits.as_ref().get_unchecked(self.word_index) };
        let used = bit_width - self.fill;
        res = ((res << used) | (self.window >> (B::Word::BITS as usize - used))) & self.vec.mask; // not in a loop
        self.window <<= used;
        self.fill = B::Word::BITS as usize - used;
        res
    }
}

impl<'a, B: Backend<Word: Word> + AsRef<[B::Word]>> IntoUncheckedBackIterator
    for &'a BitFieldVec<B>
{
    type Item = B::Word;
    type IntoUncheckedIterBack = BitFieldVecUncheckedBackIter<'a, B>;

    fn into_unchecked_iter_back(self) -> Self::IntoUncheckedIterBack {
        BitFieldVecUncheckedBackIter::new(self, self.len())
    }

    fn into_unchecked_iter_back_from(self, from: usize) -> Self::IntoUncheckedIterBack {
        BitFieldVecUncheckedBackIter::new(self, from)
    }
}

/// An [`Iterator`] over the values of a [`BitFieldVec`].
#[derive(Debug, Clone, MemSize, MemDbg)]
pub struct BitFieldVecIter<'a, B: Backend<Word: Word>> {
    unchecked: BitFieldVecUncheckedIter<'a, B>,
    range: core::ops::Range<usize>,
}

impl<'a, B: Backend<Word: Word> + AsRef<[B::Word]>> BitFieldVecIter<'a, B> {
    fn new(vec: &'a BitFieldVec<B>, from: usize) -> Self {
        let len = vec.len();
        if from > len {
            panic!("Start index out of bounds: {} > {}", from, len);
        }
        Self {
            unchecked: BitFieldVecUncheckedIter::new(vec, from),
            range: from..len,
        }
    }
}

impl<B: Backend<Word: Word> + AsRef<[B::Word]>> Iterator for BitFieldVecIter<'_, B> {
    type Item = B::Word;
    fn next(&mut self) -> Option<Self::Item> {
        if self.range.is_empty() {
            return None;
        }
        // SAFETY: index has just been checked
        let res = unsafe { self.unchecked.next_unchecked() };
        self.range.start += 1;
        Some(res)
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.range.len(), Some(self.range.len()))
    }
}

impl<B: Backend<Word: Word> + AsRef<[B::Word]>> ExactSizeIterator for BitFieldVecIter<'_, B> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.range.len()
    }
}

impl<B: Backend<Word: Word> + AsRef<[B::Word]>> FusedIterator for BitFieldVecIter<'_, B> {}

/// This implements iteration from the end, but it's slower than the forward iteration
/// as here we do a random access, while in the forward iterator we do a sequential access
/// and we keep a buffer of `W::BITS` bits to speed up the iteration.
///
/// If needed we could also keep a buffer from the end, but the logic would be more complex
/// and potentially slower.
impl<B: Backend<Word: Word> + AsRef<[B::Word]>> DoubleEndedIterator for BitFieldVecIter<'_, B> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.range.is_empty() {
            return None;
        }
        self.range.end -= 1;
        // SAFETY: range.end was > range.start, so it is a valid index
        let res = unsafe { self.unchecked.vec.get_value_unchecked(self.range.end) };
        Some(res)
    }
}

/// Equality between bit-field vectors requires that the word is the same, the
/// bit width is the same, and the content is the same.
impl<B: Backend<Word: Word> + AsRef<[B::Word]>, C: Backend<Word = B::Word> + AsRef<[B::Word]>>
    PartialEq<BitFieldVec<C>> for BitFieldVec<B>
{
    fn eq(&self, other: &BitFieldVec<C>) -> bool {
        if self.bit_width() != other.bit_width() {
            return false;
        }
        if self.len() != other.len() {
            return false;
        }
        let bits = B::Word::BITS as usize;
        let bit_len = self.len() * self.bit_width();
        if self.bits.as_ref()[..bit_len / bits] != other.bits.as_ref()[..bit_len / bits] {
            return false;
        }

        let residual = bit_len % bits;
        residual == 0
            || (self.bits.as_ref()[bit_len / bits] ^ other.bits.as_ref()[bit_len / bits])
                << (bits - residual)
                == B::Word::ZERO
    }
}

impl<'a, B: Backend<Word: Word> + AsRef<[B::Word]>> IntoIterator for &'a BitFieldVec<B> {
    type Item = B::Word;
    type IntoIter = BitFieldVecIter<'a, B>;

    fn into_iter(self) -> Self::IntoIter {
        BitFieldVecIter::new(self, 0)
    }
}

impl<'a, B: Backend<Word: Word> + AsRef<[B::Word]>> IntoIteratorFrom for &'a BitFieldVec<B> {
    type IntoIterFrom = BitFieldVecIter<'a, B>;

    fn into_iter_from(self, from: usize) -> Self::IntoIterFrom {
        BitFieldVecIter::new(self, from)
    }
}

impl<W: Word> core::iter::Extend<W> for BitFieldVec<Vec<W>> {
    fn extend<T: IntoIterator<Item = W>>(&mut self, iter: T) {
        for value in iter {
            self.push(value);
        }
    }
}

impl<B: Backend<Word: Word> + AsRef<[B::Word]>> BitFieldVec<B> {
    pub fn iter_from(&self, from: usize) -> BitFieldVecIter<'_, B> {
        BitFieldVecIter::new(self, from)
    }

    pub fn iter(&self) -> BitFieldVecIter<'_, B> {
        self.iter_from(0)
    }
}
/// A tentatively thread-safe vector of bit fields of fixed width (AKA "compact arrays",
/// "bit array", etc.)
///
/// This implementation provides some concurrency guarantees, albeit not
/// full-fledged thread safety: more precisely, we can guarantee thread-safety
/// if the bit width is a power of two; otherwise, concurrent writes to values
/// that cross word boundaries might end up in different threads succeeding in
/// writing only part of a value. If the user can guarantee that no two threads
/// ever write to the same boundary-crossing value, then no race condition can
/// happen.
///
/// See the [module documentation](crate::bits) for more details.
#[derive(Debug, Clone, Hash, MemSize, MemDbg, Delegate)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde))]
#[cfg_attr(
    feature = "epserde",
    epserde(bound(
        deser = "for<'a> <B as epserde::deser::DeserInner>::DeserType<'a>: Backend<Word = B::Word>"
    ))
)]
#[delegate(crate::traits::Backend, target = "bits")]
pub struct AtomicBitFieldVec<
    B: Backend<Word: PrimitiveAtomicUnsigned<Value: Word>> = Vec<Atomic<usize>>,
> {
    /// The underlying storage.
    bits: B,
    /// The bit width of the values stored in the vector.
    bit_width: usize,
    /// A mask with its lowest `bit_width` bits set to one.
    mask: <B::Word as PrimitiveAtomic>::Value,
    /// The length of the vector.
    len: usize,
}

impl<B: Backend<Word: PrimitiveAtomicUnsigned<Value: Word>>> AtomicBitFieldVec<B> {
    /// # Safety
    /// `len` * `bit_width` must be between 0 (included) and the number of
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

    /// Returns the backend, the bit width, and the length, consuming this
    /// vector.
    #[inline(always)]
    pub fn into_raw_parts(self) -> (B, usize, usize) {
        (self.bits, self.bit_width, self.len)
    }

    /// Returns the mask used to extract values from the vector.
    /// This will keep the lowest `bit_width` bits.
    pub const fn mask(&self) -> <B::Word as PrimitiveAtomic>::Value {
        self.mask
    }
}

impl<B: Backend<Word: PrimitiveAtomicUnsigned<Value: Word>> + AsRef<[B::Word]>>
    AtomicBitFieldVec<B>
{
    /// Returns the backend of the `AtomicBitFieldVec` as a slice of atomic words.
    pub fn as_slice(&self) -> &[B::Word] {
        self.bits.as_ref()
    }
}

impl<A: PrimitiveAtomicUnsigned<Value: Word>> AtomicBitFieldVec<Vec<A>> {
    pub fn new(bit_width: usize, len: usize) -> AtomicBitFieldVec<Vec<A>> {
        // we need at least two words to avoid branches in the gets
        let n_of_words = Ord::max(1, (len * bit_width).div_ceil(A::Value::BITS as usize));
        AtomicBitFieldVec {
            bits: (0..n_of_words).map(|_| A::new(A::Value::ZERO)).collect(),
            bit_width,
            mask: mask(bit_width),
            len,
        }
    }
}

impl<B: Backend<Word: PrimitiveAtomicUnsigned<Value: Word>>> AtomicBitWidth
    for AtomicBitFieldVec<B>
{
    #[inline(always)]
    fn atomic_bit_width(&self) -> usize {
        debug_assert!(self.bit_width <= <B::Word as PrimitiveAtomic>::Value::BITS as usize);
        self.bit_width
    }
}

impl<B: Backend<Word: PrimitiveAtomicUnsigned<Value: Word>> + AsRef<[B::Word]>> SliceByValue
    for AtomicBitFieldVec<B>
{
    type Value = <B::Word as PrimitiveAtomic>::Value;

    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }

    unsafe fn get_value_unchecked(&self, index: usize) -> Self::Value {
        unsafe { self.get_atomic_unchecked(index, Ordering::Relaxed) }
    }
}

impl<B: Backend<Word: PrimitiveAtomicUnsigned<Value: Word>> + AsRef<[B::Word]>>
    AtomicBitFieldSlice<B::Word> for AtomicBitFieldVec<B>
{
    #[inline]
    fn len(&self) -> usize {
        self.len
    }

    #[inline]
    unsafe fn get_atomic_unchecked(
        &self,
        index: usize,
        order: Ordering,
    ) -> <B::Word as PrimitiveAtomic>::Value {
        let wbits = <B::Word as PrimitiveAtomic>::Value::BITS as usize;
        let pos = index * self.bit_width;
        let word_index = pos / wbits;
        let bit_index = pos % wbits;
        let data = self.bits.as_ref();

        unsafe {
            if bit_index + self.bit_width <= wbits {
                (data.get_unchecked(word_index).load(order) >> bit_index) & self.mask
            } else {
                ((data.get_unchecked(word_index).load(order) >> bit_index)
                    | (data.get_unchecked(word_index + 1).load(order) << (wbits - bit_index)))
                    & self.mask
            }
        }
    }

    // We reimplement set as we have the mask in the structure.

    /// Sets the element of the slice at the specified index.
    ///
    /// May panic if the index is not in [0..[len](SliceByValue::len))
    /// or the value does not fit in [`BitWidth::bit_width`] bits.
    #[inline(always)]
    fn set_atomic(
        &self,
        index: usize,
        value: <B::Word as PrimitiveAtomic>::Value,
        order: Ordering,
    ) {
        panic_if_out_of_bounds!(index, self.len);
        panic_if_value!(value, self.mask, self.bit_width);
        unsafe {
            self.set_atomic_unchecked(index, value, order);
        }
    }

    #[inline]
    unsafe fn set_atomic_unchecked(
        &self,
        index: usize,
        value: <B::Word as PrimitiveAtomic>::Value,
        order: Ordering,
    ) {
        unsafe {
            let wbits = <B::Word as PrimitiveAtomic>::Value::BITS as usize;
            debug_assert!(self.bit_width <= wbits);
            let pos = index * self.bit_width;
            let word_index = pos / wbits;
            let bit_index = pos % wbits;
            let data = self.bits.as_ref();

            if bit_index + self.bit_width <= wbits {
                // this is consistent
                let mut current = data.get_unchecked(word_index).load(order);
                loop {
                    let mut new = current;
                    new &= !(self.mask << bit_index);
                    new |= value << bit_index;

                    match data
                        .get_unchecked(word_index)
                        .compare_exchange(current, new, order, order)
                    {
                        Ok(_) => break,
                        Err(e) => current = e,
                    }
                }
            } else {
                let mut word = data.get_unchecked(word_index).load(order);
                // try to wait for the other thread to finish
                fence(Ordering::Acquire);
                loop {
                    let mut new = word;
                    new &= (<B::Word as PrimitiveAtomic>::Value::ONE << bit_index)
                        - <B::Word as PrimitiveAtomic>::Value::ONE;
                    new |= value << bit_index;

                    match data
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
                // should try to synchronize the threads as much as possible
                compiler_fence(Ordering::SeqCst);

                let mut word = data.get_unchecked(word_index + 1).load(order);
                fence(Ordering::Acquire);
                loop {
                    let mut new = word;
                    new &= !(self.mask >> (wbits - bit_index));
                    new |= value >> (wbits - bit_index);

                    match data
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
    }

    fn reset_atomic(&mut self, ordering: Ordering) {
        let bit_len = self.len * self.bit_width;
        let full_words = bit_len / <B::Word as PrimitiveAtomic>::Value::BITS as usize;
        let residual = bit_len % <B::Word as PrimitiveAtomic>::Value::BITS as usize;
        let bits = self.bits.as_ref();
        bits[..full_words]
            .iter()
            .for_each(|x| x.store(<B::Word as PrimitiveAtomic>::Value::ZERO, ordering));
        if residual != 0 {
            bits[full_words].fetch_and(
                <B::Word as PrimitiveAtomic>::Value::MAX << residual,
                ordering,
            );
        }
    }

    #[cfg(feature = "rayon")]
    fn par_reset_atomic(&mut self, ordering: Ordering) {
        let bit_len = self.len * self.bit_width;
        let full_words = bit_len / <B::Word as PrimitiveAtomic>::Value::BITS as usize;
        let residual = bit_len % <B::Word as PrimitiveAtomic>::Value::BITS as usize;
        let bits = self.bits.as_ref();
        bits[..full_words]
            .par_iter()
            .with_min_len(crate::RAYON_MIN_LEN)
            .for_each(|x| x.store(<B::Word as PrimitiveAtomic>::Value::ZERO, ordering));
        if residual != 0 {
            bits[full_words].fetch_and(
                <B::Word as PrimitiveAtomic>::Value::MAX << residual,
                ordering,
            );
        }
    }
}

// Conversions

impl<W: Word + AtomicPrimitive<Atomic: PrimitiveAtomicUnsigned>>
    From<AtomicBitFieldVec<Vec<W::Atomic>>> for BitFieldVec<Vec<W>>
{
    #[inline]
    fn from(value: AtomicBitFieldVec<Vec<W::Atomic>>) -> Self {
        BitFieldVec {
            bits: transmute_vec_from_atomic(value.bits),
            len: value.len,
            bit_width: value.bit_width,
            mask: value.mask,
        }
    }
}

impl<W: Word + AtomicPrimitive<Atomic: PrimitiveAtomicUnsigned>>
    From<AtomicBitFieldVec<Box<[W::Atomic]>>> for BitFieldVec<Box<[W]>>
{
    #[inline]
    fn from(value: AtomicBitFieldVec<Box<[W::Atomic]>>) -> Self {
        BitFieldVec {
            bits: transmute_boxed_slice_from_atomic(value.bits),

            len: value.len,
            bit_width: value.bit_width,
            mask: value.mask,
        }
    }
}

impl<'a, W: Word + AtomicPrimitive<Atomic: PrimitiveAtomicUnsigned>>
    From<AtomicBitFieldVec<&'a [W::Atomic]>> for BitFieldVec<&'a [W]>
{
    #[inline]
    fn from(value: AtomicBitFieldVec<&'a [W::Atomic]>) -> Self {
        BitFieldVec {
            bits: unsafe { core::mem::transmute::<&'a [W::Atomic], &'a [W]>(value.bits) },
            len: value.len,
            bit_width: value.bit_width,
            mask: value.mask,
        }
    }
}

impl<'a, W: Word + AtomicPrimitive<Atomic: PrimitiveAtomicUnsigned>>
    From<AtomicBitFieldVec<&'a mut [W::Atomic]>> for BitFieldVec<&'a mut [W]>
{
    #[inline]
    fn from(value: AtomicBitFieldVec<&'a mut [W::Atomic]>) -> Self {
        BitFieldVec {
            bits: unsafe { core::mem::transmute::<&'a mut [W::Atomic], &'a mut [W]>(value.bits) },
            len: value.len,
            bit_width: value.bit_width,
            mask: value.mask,
        }
    }
}

impl<W: Word + AtomicPrimitive<Atomic: PrimitiveAtomicUnsigned>> From<BitFieldVec<Vec<W>>>
    for AtomicBitFieldVec<Vec<W::Atomic>>
{
    #[inline]
    fn from(value: BitFieldVec<Vec<W>>) -> Self {
        AtomicBitFieldVec {
            bits: transmute_vec_into_atomic(value.bits),

            len: value.len,
            bit_width: value.bit_width,
            mask: value.mask,
        }
    }
}

impl<W: Word + AtomicPrimitive<Atomic: PrimitiveAtomicUnsigned>> From<BitFieldVec<Box<[W]>>>
    for AtomicBitFieldVec<Box<[W::Atomic]>>
{
    #[inline]
    fn from(value: BitFieldVec<Box<[W]>>) -> Self {
        AtomicBitFieldVec {
            bits: transmute_boxed_slice_into_atomic(value.bits),
            len: value.len,
            bit_width: value.bit_width,
            mask: value.mask,
        }
    }
}

impl<'a, W: Word + AtomicPrimitive<Atomic: PrimitiveAtomicUnsigned>> TryFrom<BitFieldVec<&'a [W]>>
    for AtomicBitFieldVec<&'a [W::Atomic]>
{
    type Error = CannotCastToAtomicError<W>;

    #[inline]
    fn try_from(value: BitFieldVec<&'a [W]>) -> Result<Self, Self::Error> {
        if core::mem::align_of::<W::Atomic>() != core::mem::align_of::<W>() {
            return Err(CannotCastToAtomicError::default());
        }
        Ok(AtomicBitFieldVec {
            bits: unsafe { core::mem::transmute::<&'a [W], &'a [W::Atomic]>(value.bits) },
            len: value.len,
            bit_width: value.bit_width,
            mask: value.mask,
        })
    }
}

impl<'a, W: Word + AtomicPrimitive<Atomic: PrimitiveAtomicUnsigned>>
    TryFrom<BitFieldVec<&'a mut [W]>> for AtomicBitFieldVec<&'a mut [W::Atomic]>
{
    type Error = CannotCastToAtomicError<W>;

    #[inline]
    fn try_from(value: BitFieldVec<&'a mut [W]>) -> Result<Self, Self::Error> {
        if core::mem::align_of::<W::Atomic>() != core::mem::align_of::<W>() {
            return Err(CannotCastToAtomicError::default());
        }
        Ok(AtomicBitFieldVec {
            bits: unsafe { core::mem::transmute::<&'a mut [W], &'a mut [W::Atomic]>(value.bits) },
            len: value.len,
            bit_width: value.bit_width,
            mask: value.mask,
        })
    }
}

impl<W: Word> From<BitFieldVec<Vec<W>>> for BitFieldVec<Box<[W]>> {
    fn from(value: BitFieldVec<Vec<W>>) -> Self {
        BitFieldVec {
            bits: value.bits.into_boxed_slice(),
            len: value.len,
            bit_width: value.bit_width,
            mask: value.mask,
        }
    }
}

impl<W: Word> From<BitFieldVec<Box<[W]>>> for BitFieldVec<Vec<W>> {
    fn from(value: BitFieldVec<Box<[W]>>) -> Self {
        BitFieldVec {
            bits: value.bits.into_vec(),
            len: value.len,
            bit_width: value.bit_width,
            mask: value.mask,
        }
    }
}

impl<'a, B: Backend<Word: Word> + AsRef<[B::Word]>> value_traits::iter::IterateByValueGat<'a>
    for BitFieldVec<B>
{
    type Item = B::Word;
    type Iter = BitFieldVecIter<'a, B>;
}

impl<B: Backend<Word: Word> + AsRef<[B::Word]>> value_traits::iter::IterateByValue
    for BitFieldVec<B>
{
    fn iter_value(&self) -> <Self as value_traits::iter::IterateByValueGat<'_>>::Iter {
        self.iter_from(0)
    }
}

impl<'a, B: Backend<Word: Word> + AsRef<[B::Word]>> value_traits::iter::IterateByValueFromGat<'a>
    for BitFieldVec<B>
{
    type Item = B::Word;
    type IterFrom = BitFieldVecIter<'a, B>;
}

impl<B: Backend<Word: Word> + AsRef<[B::Word]>> value_traits::iter::IterateByValueFrom
    for BitFieldVec<B>
{
    fn iter_value_from(
        &self,
        from: usize,
    ) -> <Self as value_traits::iter::IterateByValueGat<'_>>::Iter {
        self.iter_from(from)
    }
}

impl<'a, 'b, B: Backend<Word: Word> + AsRef<[B::Word]>> value_traits::iter::IterateByValueGat<'a>
    for BitFieldVecSubsliceImpl<'b, B>
{
    type Item = B::Word;
    type Iter = BitFieldVecIter<'a, B>;
}

impl<'a, B: Backend<Word: Word> + AsRef<[B::Word]>> value_traits::iter::IterateByValue
    for BitFieldVecSubsliceImpl<'a, B>
{
    fn iter_value(&self) -> <Self as value_traits::iter::IterateByValueGat<'_>>::Iter {
        self.slice.iter_from(0)
    }
}

impl<'a, 'b, B: Backend<Word: Word> + AsRef<[B::Word]>>
    value_traits::iter::IterateByValueFromGat<'a> for BitFieldVecSubsliceImpl<'b, B>
{
    type Item = B::Word;
    type IterFrom = BitFieldVecIter<'a, B>;
}

impl<'a, B: Backend<Word: Word> + AsRef<[B::Word]>> value_traits::iter::IterateByValueFrom
    for BitFieldVecSubsliceImpl<'a, B>
{
    fn iter_value_from(
        &self,
        from: usize,
    ) -> <Self as value_traits::iter::IterateByValueGat<'_>>::Iter {
        self.slice.iter_from(from)
    }
}

impl<'a, 'b, B: Backend<Word: Word> + AsRef<[B::Word]>> value_traits::iter::IterateByValueGat<'a>
    for BitFieldVecSubsliceImplMut<'b, B>
{
    type Item = B::Word;
    type Iter = BitFieldVecIter<'a, B>;
}

impl<'a, B: Backend<Word: Word> + AsRef<[B::Word]>> value_traits::iter::IterateByValue
    for BitFieldVecSubsliceImplMut<'a, B>
{
    fn iter_value(&self) -> <Self as value_traits::iter::IterateByValueGat<'_>>::Iter {
        self.slice.iter_from(0)
    }
}

impl<'a, 'b, B: Backend<Word: Word> + AsRef<[B::Word]>>
    value_traits::iter::IterateByValueFromGat<'a> for BitFieldVecSubsliceImplMut<'b, B>
{
    type Item = B::Word;
    type IterFrom = BitFieldVecIter<'a, B>;
}

impl<'a, B: Backend<Word: Word> + AsRef<[B::Word]>> value_traits::iter::IterateByValueFrom
    for BitFieldVecSubsliceImplMut<'a, B>
{
    fn iter_value_from(
        &self,
        from: usize,
    ) -> <Self as value_traits::iter::IterateByValueGat<'_>>::Iter {
        self.slice.iter_from(from)
    }
}

/// A transparent wrapper around [`BitFieldVec`] that implements [`SliceByValue`]
/// using unaligned reads.
///
/// This wrapper delegates [`SliceByValue::get_value_unchecked`] to
/// [`BitFieldVec::get_unaligned_unchecked`], which can be faster for random
/// access patterns.
///
/// The [`TryIntoUnaligned`](crate::traits::TryIntoUnaligned) trait converts a
/// [`BitFieldVec`] into an [`BitFieldVecU`] after adding a padding word at the
/// end, which is required for unaligned reads to work correctly. The conversion
/// will fail if the bit width does not satisfy the constraints of
/// [`BitFieldVec::get_unaligned_unchecked`]. You can recover the original
/// [`BitFieldVec`] using a [`From`
/// implementation](#impl-From<BitFieldVecU<Box<%5BW%5D>>>-for-BitFieldVec<Box<%5BW%5D>>).
///
/// # Safety
///
/// The backing storage must have sufficient padding at the end so that
/// unaligned reads do not access memory outside the allocation.
#[derive(Debug, Clone, MemSize, MemDbg, value_traits::Subslices)]
#[value_traits_subslices(bound = "B: AsRef<[B::Word]>")]
#[value_traits_subslices(bound = "B::Word: Word")]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde))]
#[repr(transparent)]
#[cfg_attr(
    feature = "epserde",
    epserde(bound(
        deser = "B: Backend + epserde::deser::DeserInner, for<'a> <B as epserde::deser::DeserInner>::DeserType<'a>: Backend<Word = B::Word>"
    ))
)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "B: Backend + serde::Serialize, B::Word: serde::Serialize",
        deserialize = "B: Backend + serde::Deserialize<'de>, B::Word: serde::Deserialize<'de>"
    ))
)]
pub struct BitFieldVecU<B: Backend<Word: Word> = Vec<usize>>(BitFieldVec<B>);

impl<B: Backend<Word: Word>> BitFieldVecU<B> {
    /// Returns the mask used to extract values from the vector.
    /// This will keep the lowest `bit_width` bits.
    pub const fn mask(&self) -> B::Word {
        self.0.mask()
    }
}

impl<B: Backend<Word: Word>> Backend for BitFieldVecU<B> {
    type Word = B::Word;
}

impl<B: Backend<Word: Word>> BitWidth for BitFieldVecU<B> {
    #[inline(always)]
    fn bit_width(&self) -> usize {
        self.0.bit_width()
    }
}

impl<B: Backend<Word: Word> + AsRef<[B::Word]>> BitFieldSlice for BitFieldVecU<B> {
    fn as_slice(&self) -> &[Self::Value] {
        self.0.as_slice()
    }
}

impl<B: Backend<Word: Word> + AsRef<[B::Word]>> AsRef<[B::Word]> for BitFieldVecU<B> {
    fn as_ref(&self) -> &[B::Word] {
        self.0.bits.as_ref()
    }
}

impl<W: Word> crate::traits::TryIntoUnaligned for Box<[W]> {
    type Unaligned = Box<[W]>;
    fn try_into_unaligned(
        self,
    ) -> Result<Self::Unaligned, crate::traits::UnalignedConversionError> {
        Ok(self)
    }
}

impl<W: Word> crate::traits::TryIntoUnaligned for Vec<W> {
    type Unaligned = Vec<W>;
    fn try_into_unaligned(
        self,
    ) -> Result<Self::Unaligned, crate::traits::UnalignedConversionError> {
        Ok(self)
    }
}

impl<W: Word, const N: usize> crate::traits::TryIntoUnaligned for [W; N] {
    type Unaligned = [W; N];
    fn try_into_unaligned(
        self,
    ) -> Result<Self::Unaligned, crate::traits::UnalignedConversionError> {
        Ok(self)
    }
}

impl<W: Word> crate::traits::TryIntoUnaligned for BitFieldVec<Box<[W]>> {
    type Unaligned = BitFieldVecU<Box<[W]>>;

    /// Converts a [`BitFieldVec`] into a [`BitFieldVecU`], adding a
    /// padding word at the end if one is not already present.
    ///
    /// # Errors
    ///
    /// Returns an error if the bit width does not satisfy the constraints of
    /// [`BitFieldVec::get_unaligned_unchecked`]: it must be at most
    /// `W::BITS - 6`, or exactly `W::BITS - 4`, or exactly `W::BITS`.
    fn try_into_unaligned(
        self,
    ) -> Result<Self::Unaligned, crate::traits::UnalignedConversionError> {
        let bw = self.bit_width();
        ensure_unaligned!(W, bw);
        let needed = (SliceByValue::len(&self) * bw).div_ceil(W::BITS as usize);
        if self.as_slice().len() > needed {
            // Padding word already present (e.g., built with new_padded).
            Ok(BitFieldVecU(self))
        } else {
            // Add a padding word, reserving exactly one extra slot to
            // avoid over-allocation.
            let (raw_bits, bit_width, len) = self.into_raw_parts();
            let mut v = raw_bits.into_vec();
            v.reserve_exact(1);
            v.push(W::ZERO);
            // SAFETY: we added a padding word, the length is still valid
            Ok(BitFieldVecU(unsafe {
                BitFieldVec::from_raw_parts(v.into_boxed_slice(), bit_width, len)
            }))
        }
    }
}

impl<W: Word> From<BitFieldVecU<Box<[W]>>> for BitFieldVec<Box<[W]>> {
    /// Converts a [`BitFieldVecU`] back into a [`BitFieldVec`].
    ///
    /// The padding word is kept in the backing storage so that a
    /// subsequent [`try_into_unaligned`](crate::traits::TryIntoUnaligned::try_into_unaligned)
    /// does not need to reallocate.
    fn from(unaligned: BitFieldVecU<Box<[W]>>) -> Self {
        unaligned.0
    }
}

impl<B: Backend<Word: Word> + AsRef<[B::Word]>> SliceByValue for BitFieldVecU<B> {
    type Value = B::Word;

    #[inline(always)]
    fn len(&self) -> usize {
        SliceByValue::len(&self.0)
    }

    #[inline(always)]
    unsafe fn get_value_unchecked(&self, index: usize) -> B::Word {
        unsafe { self.0.get_unaligned_unchecked(index) }
    }
}

impl<'a, B: Backend<Word: Word> + AsRef<[B::Word]>> IntoUncheckedIterator for &'a BitFieldVecU<B> {
    type Item = B::Word;
    type IntoUncheckedIter = BitFieldVecUncheckedIter<'a, B>;
    fn into_unchecked_iter_from(self, from: usize) -> Self::IntoUncheckedIter {
        BitFieldVecUncheckedIter::new(&self.0, from)
    }
}

impl<'a, B: Backend<Word: Word> + AsRef<[B::Word]>> IntoUncheckedBackIterator
    for &'a BitFieldVecU<B>
{
    type Item = B::Word;
    type IntoUncheckedIterBack = BitFieldVecUncheckedBackIter<'a, B>;

    fn into_unchecked_iter_back(self) -> Self::IntoUncheckedIterBack {
        BitFieldVecUncheckedBackIter::new(&self.0, SliceByValue::len(&self.0))
    }

    fn into_unchecked_iter_back_from(self, from: usize) -> Self::IntoUncheckedIterBack {
        BitFieldVecUncheckedBackIter::new(&self.0, from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_with_capacity() {
        let mut b = BitFieldVec::<Vec<usize>>::with_capacity(10, 100);
        let capacity = b.bits.capacity();
        for _ in 0..100 {
            b.push(0);
        }
        assert_eq!(b.bits.capacity(), capacity);
    }

    fn copy<
        B: Backend<Word: Word> + AsRef<[B::Word]>,
        C: Backend<Word = B::Word> + AsRef<[B::Word]> + AsMut<[B::Word]>,
    >(
        source: &BitFieldVec<B>,
        from: usize,
        dest: &mut BitFieldVec<C>,
        to: usize,
        len: usize,
    ) {
        let len = Ord::min(Ord::min(len, dest.len - to), source.len - from);
        for i in 0..len {
            dest.set_value(to + i, source.index_value(from + i));
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
