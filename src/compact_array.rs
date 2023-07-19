use crate::traits::*;
use anyhow::Result;
use std::{
    io::{Seek, Write},
    sync::atomic::{compiler_fence, fence, AtomicU64, Ordering},
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CompactArray<B> {
    data: B,
    bit_width: usize,
    len: usize,
}

impl CompactArray<Vec<u64>> {
    pub fn new(bit_width: usize, len: usize) -> Self {
        #[cfg(not(any(feature = "testless_read", feature = "testless_write")))]
        // we need at least two words to avoid branches in the gets
        let n_of_words = (len * bit_width + 63) / 64;
        #[cfg(any(feature = "testless_read", feature = "testless_write"))]
        // we need at least two words to avoid branches in the gets
        let n_of_words = (1 + (len * bit_width + 63) / 64).max(2);
        Self {
            data: vec![0; n_of_words],
            bit_width,
            len,
        }
    }
}

impl CompactArray<Vec<AtomicU64>> {
    pub fn new_atomic(bit_width: usize, len: usize) -> Self {
        #[cfg(not(any(feature = "testless_read", feature = "testless_write")))]
        // we need at least two words to avoid branches in the gets
        let n_of_words = (len * bit_width + 63) / 64;
        #[cfg(any(feature = "testless_read", feature = "testless_write"))]
        // we need at least two words to avoid branches in the gets
        let n_of_words = (1 + (len * bit_width + 63) / 64).max(2);
        Self {
            data: (0..n_of_words).map(|_| AtomicU64::new(0)).collect(),
            bit_width,
            len,
        }
    }
}

impl<B> CompactArray<B> {
    /// # Safety
    /// TODO: this function is never used.
    #[inline(always)]
    pub unsafe fn from_raw_parts(data: B, bit_width: usize, len: usize) -> Self {
        Self {
            data,
            bit_width,
            len,
        }
    }

    #[inline(always)]
    pub fn into_raw_parts(self) -> (B, usize, usize) {
        (self.data, self.bit_width, self.len)
    }
}

impl<B: VSliceCore> VSliceCore for CompactArray<B> {
    #[inline(always)]
    fn bit_width(&self) -> usize {
        debug_assert!(self.bit_width <= self.data.bit_width());
        self.bit_width
    }

    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }
}

impl<B: VSlice> VSlice for CompactArray<B> {
    #[inline]
    unsafe fn get_unchecked(&self, index: usize) -> u64 {
        debug_assert!(self.bit_width != 64);
        #[cfg(not(feature = "testless_read"))]
        if self.bit_width == 0 {
            return 0;
        }

        let pos = index * self.bit_width;
        let word_index = pos / 64;
        let bit_index = pos % 64;

        #[cfg(feature = "testless_read")]
        {
            // ALERT!: this is not the correct mask for width 64
            let mask = (1_u64 << self.bit_width) - 1;

            let lower = self.data.get_unchecked(word_index) >> bit_index;
            let higher = (self.data.get_unchecked(word_index + 1) << (63 - bit_index)) << 1;

            (higher | lower) & mask
        }

        #[cfg(not(feature = "testless_read"))]
        {
            let l = 64 - self.bit_width;

            if bit_index <= l {
                self.data.get_unchecked(word_index) << (l - bit_index) >> l
            } else {
                self.data.get_unchecked(word_index) >> bit_index
                    | self.data.get_unchecked(word_index + 1) << (64 + l - bit_index) >> l
            }
        }
    }
}

impl<B: VSliceMutAtomicCmpExchange> VSliceAtomic for CompactArray<B> {
    #[inline]
    unsafe fn get_atomic_unchecked(&self, index: usize, order: Ordering) -> u64 {
        debug_assert!(self.bit_width != 64);
        if self.bit_width == 0 {
            return 0;
        }

        let pos = index * self.bit_width;
        let word_index = pos / 64;
        let bit_index = pos % 64;

        let l = 64 - self.bit_width;
        // we always use the tested to reduce the probability of unconsistent reads
        if bit_index <= l {
            self.data.get_atomic_unchecked(word_index, order) << (l - bit_index) >> l
        } else {
            self.data.get_atomic_unchecked(word_index, order) >> bit_index
                | self.data.get_atomic_unchecked(word_index + 1, order) << (64 + l - bit_index) >> l
        }
    }
    #[inline]
    unsafe fn set_atomic_unchecked(&self, index: usize, value: u64, order: Ordering) {
        debug_assert!(self.bit_width != 64);
        if self.bit_width == 0 {
            return;
        }

        let pos = index * self.bit_width;
        let word_index = pos / 64;
        let bit_index = pos % 64;

        let mask: u64 = (1_u64 << self.bit_width) - 1;

        let end_word_index = (pos + self.bit_width - 1) / 64;
        if word_index == end_word_index {
            // this is consistent
            let mut current = self.data.get_atomic_unchecked(word_index, order);
            loop {
                let mut new = current;
                new &= !(mask << bit_index);
                new |= value << bit_index;

                match self
                    .data
                    .compare_exchange(word_index, current, new, order, order)
                {
                    Ok(_) => break,
                    Err(e) => current = e,
                }
            }
        } else {
            // try to wait for the other thread to finish
            let mut word = self.data.get_atomic_unchecked(word_index, order);
            fence(Ordering::Acquire);
            loop {
                let mut new = word;
                new &= (1 << bit_index) - 1;
                new |= value << bit_index;

                match self
                    .data
                    .compare_exchange(word_index, word, new, order, order)
                {
                    Ok(_) => break,
                    Err(e) => word = e,
                }
            }
            fence(Ordering::Release);

            // ensure that the compiler does not reorder the two atomic operations
            // this should increase the probability of having consistency
            // between two concurrent writes as they will both execute the set
            // of the bits in the same order, and the release / acquire fence
            // should try to syncronize the threads as much as possible
            compiler_fence(Ordering::SeqCst);

            let mut word = self.data.get_atomic_unchecked(end_word_index, order);
            fence(Ordering::Acquire);
            loop {
                let mut new = word;
                new &= !(mask >> (64 - bit_index));
                new |= value >> (64 - bit_index);

                match self
                    .data
                    .compare_exchange(end_word_index, word, new, order, order)
                {
                    Ok(_) => break,
                    Err(e) => word = e,
                }
            }
            fence(Ordering::Release);
        }
    }
}

impl<B: VSliceMut> VSliceMut for CompactArray<B> {
    #[inline]
    unsafe fn set_unchecked(&mut self, index: usize, value: u64) {
        debug_assert!(self.bit_width != 64);
        #[cfg(not(feature = "testless_write"))]
        if self.bit_width == 0 {
            return;
        }

        let pos = index * self.bit_width;
        let word_index = pos / 64;
        let bit_index = pos % 64;

        let mask: u64 = (1_u64 << self.bit_width) - 1;

        #[cfg(feature = "testless_write")]
        {
            let lower = value << bit_index;
            let higher = (value >> (63 - bit_index)) >> 1;

            let lower_word = self.data.get_unchecked(word_index) & !(mask << bit_index);
            self.data.set_unchecked(word_index, lower_word | lower);

            let higher_word =
                self.data.get_unchecked(word_index + 1) & !((mask >> (63 - bit_index)) >> 1);
            self.data
                .set_unchecked(word_index + 1, higher_word | higher);
        }

        #[cfg(not(feature = "testless_write"))]
        {
            let end_word_index = (pos + self.bit_width - 1) / 64;
            if word_index == end_word_index {
                let mut word = self.data.get_unchecked(word_index);
                word &= !(mask << bit_index);
                word |= value << bit_index;
                self.data.set_unchecked(word_index, word);
            } else {
                let mut word = self.data.get_unchecked(word_index);
                word &= (1 << bit_index) - 1;
                word |= value << bit_index;
                self.data.set_unchecked(word_index, word);

                let mut word = self.data.get_unchecked(end_word_index);
                word &= !(mask >> (64 - bit_index));
                word |= value >> (64 - bit_index);
                self.data.set_unchecked(end_word_index, word);
            }
        }
    }
}

impl<B, D> ConvertTo<CompactArray<D>> for CompactArray<B>
where
    B: ConvertTo<D> + VSlice,
    D: VSlice,
{
    fn convert_to(self) -> Result<CompactArray<D>> {
        Ok(CompactArray {
            len: self.len,
            bit_width: self.bit_width,
            data: self.data.convert_to()?,
        })
    }
}

impl<B: VSlice> ConvertTo<Vec<u64>> for CompactArray<B> {
    fn convert_to(self) -> Result<Vec<u64>> {
        Ok((0..self.len())
            .map(|i| unsafe { self.get_unchecked(i) })
            .collect::<Vec<_>>())
    }
}

impl<B: VSlice + Serialize> Serialize for CompactArray<B> {
    fn serialize<F: Write + Seek>(&self, backend: &mut F) -> Result<usize> {
        let mut bytes = 0;
        bytes += self.len.serialize(backend)?;
        bytes += self.bit_width.serialize(backend)?;
        bytes += self.data.serialize(backend)?;
        Ok(bytes)
    }
}

impl<'a, B: VSlice + Deserialize<'a>> Deserialize<'a> for CompactArray<B> {
    fn deserialize(backend: &'a [u8]) -> Result<(Self, &'a [u8])> {
        let (len, backend) = usize::deserialize(backend)?;
        let (bit_width, backend) = usize::deserialize(backend)?;
        let (data, backend) = B::deserialize(backend)?;
        Ok((
            Self {
                len,
                bit_width,
                data,
            },
            backend,
        ))
    }
}

impl<B: VSlice + MemSize> MemSize for CompactArray<B> {
    fn mem_size(&self) -> usize {
        self.len.mem_size() + self.bit_width.mem_size() + self.data.mem_size()
    }
    fn mem_used(&self) -> usize {
        self.len.mem_used() + self.bit_width.mem_used() + self.data.mem_used()
    }
}

impl From<CompactArray<Vec<u64>>> for CompactArray<Vec<AtomicU64>> {
    #[inline]
    fn from(bm: CompactArray<Vec<u64>>) -> Self {
        let data = unsafe { std::mem::transmute::<Vec<u64>, Vec<AtomicU64>>(bm.data) };
        CompactArray {
            data,
            len: bm.len,
            bit_width: bm.bit_width,
        }
    }
}

impl From<CompactArray<Vec<AtomicU64>>> for CompactArray<Vec<u64>> {
    #[inline]
    fn from(bm: CompactArray<Vec<AtomicU64>>) -> Self {
        let data = unsafe { std::mem::transmute::<Vec<AtomicU64>, Vec<u64>>(bm.data) };
        CompactArray {
            data,
            len: bm.len,
            bit_width: bm.bit_width,
        }
    }
}

impl<'a> From<CompactArray<&'a [AtomicU64]>> for CompactArray<&'a [u64]> {
    #[inline]
    fn from(bm: CompactArray<&'a [AtomicU64]>) -> Self {
        let data = unsafe { std::mem::transmute::<&'a [AtomicU64], &'a [u64]>(bm.data) };
        CompactArray {
            data,
            len: bm.len,
            bit_width: bm.bit_width,
        }
    }
}

impl<'a> From<CompactArray<&'a [u64]>> for CompactArray<&'a [AtomicU64]> {
    #[inline]
    fn from(bm: CompactArray<&'a [u64]>) -> Self {
        let data = unsafe { std::mem::transmute::<&'a [u64], &'a [AtomicU64]>(bm.data) };
        CompactArray {
            data,
            len: bm.len,
            bit_width: bm.bit_width,
        }
    }
}

impl<'a> From<CompactArray<&'a mut [AtomicU64]>> for CompactArray<&'a mut [u64]> {
    #[inline]
    fn from(bm: CompactArray<&'a mut [AtomicU64]>) -> Self {
        let data = unsafe { std::mem::transmute::<&'a mut [AtomicU64], &'a mut [u64]>(bm.data) };
        CompactArray {
            data,
            len: bm.len,
            bit_width: bm.bit_width,
        }
    }
}

impl<'a> From<CompactArray<&'a mut [u64]>> for CompactArray<&'a mut [AtomicU64]> {
    #[inline]
    fn from(bm: CompactArray<&'a mut [u64]>) -> Self {
        let data = unsafe { std::mem::transmute::<&'a mut [u64], &'a mut [AtomicU64]>(bm.data) };
        CompactArray {
            data,
            len: bm.len,
            bit_width: bm.bit_width,
        }
    }
}
