use crate::traits::*;
use crate::utils::select_in_word;
use anyhow::Result;
use std::io::{Seek, Write};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SparseZeroIndex<B: SelectZeroHinted, O: VSlice, const QUANTUM_LOG2: usize = 6> {
    bits: B,
    zeros: O,
    _marker: core::marker::PhantomData<[(); QUANTUM_LOG2]>,
}

impl<B: SelectZeroHinted, O: VSlice, const QUANTUM_LOG2: usize>
    SparseZeroIndex<B, O, QUANTUM_LOG2>
{
    /// # Safety
    /// TODO: this function is never used
    #[inline(always)]
    pub unsafe fn from_raw_parts(bits: B, zeros: O) -> Self {
        Self {
            bits,
            zeros,
            _marker: core::marker::PhantomData,
        }
    }
    #[inline(always)]
    pub fn into_raw_parts(self) -> (B, O) {
        (self.bits, self.zeros)
    }
}

impl<B: SelectZeroHinted + AsRef<[u64]>, O: VSliceMut, const QUANTUM_LOG2: usize>
    SparseZeroIndex<B, O, QUANTUM_LOG2>
{
    fn build_zeros(&mut self) -> Result<()> {
        let mut number_of_ones = 0;
        let mut next_quantum = 0;
        let mut ones_index = 0;
        for (i, mut word) in self.bits.as_ref().iter().copied().enumerate() {
            word = !word;
            let ones_in_word = word.count_ones() as u64;
            // skip the word if we can
            while number_of_ones + ones_in_word > next_quantum {
                let in_word_index = select_in_word(word, (next_quantum - number_of_ones) as usize);
                let index = (i * 64) as u64 + in_word_index as u64;
                if index >= self.len() as _ {
                    return Ok(());
                }
                self.zeros.set(ones_index, index);
                next_quantum += 1 << QUANTUM_LOG2;
                ones_index += 1;
            }

            number_of_ones += ones_in_word;
        }
        Ok(())
    }
}

/// Provide the hint to the underlying structure
impl<B: SelectZeroHinted, O: VSlice, const QUANTUM_LOG2: usize> SelectZero
    for SparseZeroIndex<B, O, QUANTUM_LOG2>
{
    #[inline(always)]
    unsafe fn select_zero_unchecked(&self, rank: usize) -> usize {
        let index = rank >> QUANTUM_LOG2;
        let pos = self.zeros.get_unchecked(index);
        let rank_at_pos = index << QUANTUM_LOG2;

        self.bits
            .select_zero_unchecked_hinted(rank, pos as usize, rank_at_pos)
    }
}

/// If the underlying implementation has select zero, forward the methods
impl<B: SelectZeroHinted + Select, O: VSlice, const QUANTUM_LOG2: usize> Select
    for SparseZeroIndex<B, O, QUANTUM_LOG2>
{
    #[inline(always)]
    fn select(&self, rank: usize) -> Option<usize> {
        self.bits.select(rank)
    }
    #[inline(always)]
    unsafe fn select_unchecked(&self, rank: usize) -> usize {
        self.bits.select_unchecked(rank)
    }
}

/// If the underlying implementation has select zero, forward the methods
impl<B: SelectZeroHinted + SelectHinted, O: VSlice, const QUANTUM_LOG2: usize> SelectHinted
    for SparseZeroIndex<B, O, QUANTUM_LOG2>
{
    #[inline(always)]
    unsafe fn select_unchecked_hinted(&self, rank: usize, pos: usize, rank_at_pos: usize) -> usize {
        self.bits.select_unchecked_hinted(rank, pos, rank_at_pos)
    }
}

/// Allow the use of multiple indices, this might not be the best way to do it
/// but it works
impl<B: SelectZeroHinted + SelectHinted, O: VSlice, const QUANTUM_LOG2: usize> SelectZeroHinted
    for SparseZeroIndex<B, O, QUANTUM_LOG2>
{
    #[inline(always)]
    unsafe fn select_zero_unchecked_hinted(
        &self,
        rank: usize,
        pos: usize,
        rank_at_pos: usize,
    ) -> usize {
        let index = rank >> QUANTUM_LOG2;
        let this_pos = self.zeros.get_unchecked(index) as usize;
        let this_rank_at_pos = index << QUANTUM_LOG2;

        // choose the best hint, as in the one with rank_at_pos closest to rank
        if rank_at_pos > this_rank_at_pos {
            self.bits
                .select_zero_unchecked_hinted(rank, pos, rank_at_pos)
        } else {
            self.bits
                .select_zero_unchecked_hinted(rank, this_pos, this_rank_at_pos)
        }
    }
}

/// Forward the lengths
impl<B: SelectZeroHinted, O: VSlice, const QUANTUM_LOG2: usize> BitLength
    for SparseZeroIndex<B, O, QUANTUM_LOG2>
{
    #[inline(always)]
    fn len(&self) -> usize {
        self.bits.len()
    }
    #[inline(always)]
    fn count(&self) -> usize {
        self.bits.count()
    }
}

impl<B: SelectZeroHinted, const QUANTUM_LOG2: usize> ConvertTo<B>
    for SparseZeroIndex<B, Vec<u64>, QUANTUM_LOG2>
{
    #[inline(always)]
    fn convert_to(self) -> Result<B> {
        Ok(self.bits)
    }
}

impl<B: SelectZeroHinted + AsRef<[u64]>, const QUANTUM_LOG2: usize>
    ConvertTo<SparseZeroIndex<B, Vec<u64>, QUANTUM_LOG2>> for B
{
    #[inline(always)]
    fn convert_to(self) -> Result<SparseZeroIndex<B, Vec<u64>, QUANTUM_LOG2>> {
        let mut res = SparseZeroIndex {
            zeros: vec![0; (self.len() - self.count() + (1 << QUANTUM_LOG2) - 1) >> QUANTUM_LOG2],
            bits: self,
            _marker: core::marker::PhantomData,
        };
        res.build_zeros()?;
        Ok(res)
    }
}

impl<B, O, const QUANTUM_LOG2: usize> AsRef<[u64]> for SparseZeroIndex<B, O, QUANTUM_LOG2>
where
    B: AsRef<[u64]> + SelectZeroHinted,
    O: VSlice,
{
    fn as_ref(&self) -> &[u64] {
        self.bits.as_ref()
    }
}

impl<B: SelectZeroHinted + Serialize, O: VSlice + Serialize, const QUANTUM_LOG2: usize> Serialize
    for SparseZeroIndex<B, O, QUANTUM_LOG2>
{
    fn serialize<F: Write + Seek>(&self, backend: &mut F) -> Result<usize> {
        let mut bytes = 0;
        bytes += self.bits.serialize(backend)?;
        bytes += self.zeros.serialize(backend)?;
        Ok(bytes)
    }
}

impl<
        'a,
        B: SelectZeroHinted + Deserialize<'a>,
        O: VSlice + Deserialize<'a>,
        const QUANTUM_LOG2: usize,
    > Deserialize<'a> for SparseZeroIndex<B, O, QUANTUM_LOG2>
{
    fn deserialize(backend: &'a [u8]) -> Result<(Self, &'a [u8])> {
        let (bits, backend) = B::deserialize(backend)?;
        let (zeros, backend) = O::deserialize(backend)?;

        Ok((
            Self {
                bits,
                zeros,
                _marker: Default::default(),
            },
            backend,
        ))
    }
}

impl<B: SelectZeroHinted + MemSize, O: VSlice + MemSize, const QUANTUM_LOG2: usize> MemSize
    for SparseZeroIndex<B, O, QUANTUM_LOG2>
{
    fn mem_size(&self) -> usize {
        self.bits.mem_size() + self.zeros.mem_size()
    }
    fn mem_used(&self) -> usize {
        self.bits.mem_used() + self.zeros.mem_used()
    }
}
