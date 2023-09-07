use crate::traits::*;
use anyhow::Result;
use common_traits::SelectInWord;
use epserde::*;

#[derive(Epserde, Debug, Clone, PartialEq, Eq, Hash)]
pub struct SparseIndex<B: SelectHinted, O: VSlice, const QUANTUM_LOG2: usize = 6> {
    bits: B,
    ones: O,
    _marker: core::marker::PhantomData<[(); QUANTUM_LOG2]>,
}

impl<B: SelectHinted, O: VSlice, const QUANTUM_LOG2: usize> SparseIndex<B, O, QUANTUM_LOG2> {
    /// # Safety
    /// TODO: this function is never used
    #[inline(always)]
    pub unsafe fn from_raw_parts(bits: B, ones: O) -> Self {
        Self {
            bits,
            ones,
            _marker: core::marker::PhantomData,
        }
    }
    #[inline(always)]
    pub fn into_raw_parts(self) -> (B, O) {
        (self.bits, self.ones)
    }
}

impl<B: SelectHinted + AsRef<[u64]>, O: VSliceMut, const QUANTUM_LOG2: usize>
    SparseIndex<B, O, QUANTUM_LOG2>
{
    fn build_ones(&mut self) -> Result<()> {
        let mut number_of_ones = 0;
        let mut next_quantum = 0;
        let mut ones_index = 0;

        for (i, word) in self.bits.as_ref().iter().copied().enumerate() {
            let ones_in_word = word.count_ones() as u64;
            // skip the word if we can
            while number_of_ones + ones_in_word > next_quantum {
                let in_word_index = word.select_in_word((next_quantum - number_of_ones) as usize);
                let index = (i * 64) as u64 + in_word_index as u64;
                self.ones.set(ones_index, index);
                next_quantum += 1 << QUANTUM_LOG2;
                ones_index += 1;
            }

            number_of_ones += ones_in_word;
        }
        Ok(())
    }
}

/// Provide the hint to the underlying structure
impl<B: SelectHinted, O: VSlice, const QUANTUM_LOG2: usize> Select
    for SparseIndex<B, O, QUANTUM_LOG2>
{
    #[inline(always)]
    unsafe fn select_unchecked(&self, rank: usize) -> usize {
        let index = rank >> QUANTUM_LOG2;
        let pos = self.ones.get_unchecked(index);
        let rank_at_pos = index << QUANTUM_LOG2;

        self.bits
            .select_unchecked_hinted(rank, pos as usize, rank_at_pos)
    }
}

/// If the underlying implementation has select zero, forward the methods
impl<B: SelectHinted + SelectZero, O: VSlice, const QUANTUM_LOG2: usize> SelectZero
    for SparseIndex<B, O, QUANTUM_LOG2>
{
    #[inline(always)]
    fn select_zero(&self, rank: usize) -> Option<usize> {
        self.bits.select_zero(rank)
    }
    #[inline(always)]
    unsafe fn select_zero_unchecked(&self, rank: usize) -> usize {
        self.bits.select_zero_unchecked(rank)
    }
}

/// If the underlying implementation has select zero, forward the methods
impl<B: SelectHinted + SelectZeroHinted, O: VSlice, const QUANTUM_LOG2: usize> SelectZeroHinted
    for SparseIndex<B, O, QUANTUM_LOG2>
{
    #[inline(always)]
    unsafe fn select_zero_unchecked_hinted(
        &self,
        rank: usize,
        pos: usize,
        rank_at_pos: usize,
    ) -> usize {
        self.bits
            .select_zero_unchecked_hinted(rank, pos, rank_at_pos)
    }
}

/// Allow the use of multiple indices, this might not be the best way to do it
/// but it works
impl<B: SelectHinted + SelectZero, O: VSlice, const QUANTUM_LOG2: usize> SelectHinted
    for SparseIndex<B, O, QUANTUM_LOG2>
{
    #[inline(always)]
    unsafe fn select_unchecked_hinted(&self, rank: usize, pos: usize, rank_at_pos: usize) -> usize {
        let index = rank >> QUANTUM_LOG2;
        let this_pos = self.ones.get_unchecked(index) as usize;
        let this_rank_at_pos = index << QUANTUM_LOG2;

        // choose the best hint, as in the one with rank_at_pos closest to rank
        if rank_at_pos > this_rank_at_pos {
            self.bits.select_unchecked_hinted(rank, pos, rank_at_pos)
        } else {
            self.bits
                .select_unchecked_hinted(rank, this_pos, this_rank_at_pos)
        }
    }
}

/// Forward the lengths
impl<B: SelectHinted + BitLength, O: VSlice, const QUANTUM_LOG2: usize> BitLength
    for SparseIndex<B, O, QUANTUM_LOG2>
{
    #[inline(always)]
    fn len(&self) -> usize {
        self.bits.len()
    }
}

impl<B: SelectHinted, O: VSlice, const QUANTUM_LOG2: usize> BitCount
    for SparseIndex<B, O, QUANTUM_LOG2>
{
    #[inline(always)]
    fn count(&self) -> usize {
        self.bits.count()
    }
}

impl<B: SelectHinted, const QUANTUM_LOG2: usize> ConvertTo<B>
    for SparseIndex<B, Vec<u64>, QUANTUM_LOG2>
{
    #[inline(always)]
    fn convert_to(self) -> Result<B> {
        Ok(self.bits)
    }
}

impl<B: SelectHinted + AsRef<[u64]>, const QUANTUM_LOG2: usize>
    ConvertTo<SparseIndex<B, Vec<u64>, QUANTUM_LOG2>> for B
{
    #[inline(always)]
    fn convert_to(self) -> Result<SparseIndex<B, Vec<u64>, QUANTUM_LOG2>> {
        let mut res = SparseIndex {
            ones: vec![0; (self.count() + (1 << QUANTUM_LOG2) - 1) >> QUANTUM_LOG2],
            bits: self,
            _marker: core::marker::PhantomData,
        };
        res.build_ones()?;
        Ok(res)
    }
}

impl<B, O, const QUANTUM_LOG2: usize> AsRef<[u64]> for SparseIndex<B, O, QUANTUM_LOG2>
where
    B: AsRef<[u64]> + SelectHinted,
    O: VSlice,
{
    fn as_ref(&self) -> &[u64] {
        self.bits.as_ref()
    }
}
