use anyhow::Result;
use crate::traits::*;

pub struct CompactArray<B: VSlice> {
    data: B,
    bit_width: usize,
    mask: u64,
    len: usize,
}

impl CompactArray<Vec<u64>> {
    pub fn new(bit_width: usize, len: usize) -> Self {
        let n_of_words = (len + bit_width - 1) / bit_width;
        Self {
            data: vec![0; n_of_words],
            bit_width,
            len,
            mask: (1 << bit_width) - 1,
        }
    }
}

impl<B: VSlice> CompactArray<B> {
    pub unsafe fn from_raw_parts(data: B, bit_width: usize, len: usize) -> Self {
        Self {
            data,
            bit_width,
            len,
            mask: (1 << bit_width) - 1,
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

    unsafe fn get_unchecked(&self, index: usize) -> u64 {
        let pos = index * self.bit_width;
        let o1 = pos & 0b11_1111;
        let o2 = 64 - o1;
    
        let mask = (1 << self.bit_width) - 1;
        let base = (pos / 64) as usize;

        let lower = (self.data.get_unchecked(base) >> o1) & mask;
        let higher = self.data.get_unchecked(base + 1) >> o2;
        
        (higher | lower) & mask
    }
}

impl<B: VSliceMut> VSliceMut for CompactArray<B> {
    unsafe fn set_unchecked(&mut self, index: usize, value: u64) {
        let pos = index * self.bit_width;
        let o1 = pos & 0b11_1111;
        let o2 = 64 - o1;
    
        let base = (pos / 64) as usize;
        let lower = value << o1;
        let higher = value >> o2;
        
        // TODO!: should this clean the previous bits?
        self.data.set_unchecked(base, self.data.get_unchecked(base) | lower);
        self.data.set_unchecked(base + 1, 
            self.data.get_unchecked(base + 1) | higher
        );
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
            mask: self.mask,
            bit_width: self.bit_width,
            data: self.data.convert_to()?,
        })
    }
}

impl<B: VSlice> ConvertTo<Vec<u64>> for CompactArray<B> {
    fn convert_to(self) -> Result<Vec<u64>> {
        Ok((0..self.len())
            .map(|i| unsafe{self.get_unchecked(i)})
            .collect::<Vec<_>>()
        )
    }
}