use crate::traits::*;
use anyhow::Result;
use std::io::{Seek, Write};

pub struct CompactArray<B: VSlice> {
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

impl<B: VSlice> CompactArray<B> {
    pub unsafe fn from_raw_parts(data: B, bit_width: usize, len: usize) -> Self {
        Self {
            data,
            bit_width,
            len,
        }
    }
}

// TODO!: add invariant that Self::bit_width() <= B::bit_width()
impl<B: VSlice> VSlice for CompactArray<B> {
    #[inline(always)]
    fn bit_width(&self) -> usize {
        self.bit_width
    }

    #[inline(always)]
    fn len(&self) -> usize {
        self.len
    }

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

impl<B: VSlice + core::fmt::Debug> core::fmt::Debug for CompactArray<B> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CompactArray")
            .field("len", &self.len)
            .field("bit_width", &self.bit_width)
            .field("data", &self.data)
            .finish()
    }
}

impl<B: VSlice + Clone> Clone for CompactArray<B> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            len: self.len,
            bit_width: self.bit_width,
        }
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
        let (len, backend) = usize::deserialize(&backend)?;
        let (bit_width, backend) = usize::deserialize(&backend)?;
        let (data, backend) = B::deserialize(&backend)?;
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
