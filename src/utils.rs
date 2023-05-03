
#[inline(always)]
pub fn select_in_word(mut word: u64, rank: usize) -> usize {
    #[cfg(target_feature="bmi2")]
    {
        use core::arch::x86_64::_pdep_u64;
        /// A Fast x86 Implementation of Select
        /// by Prashant Pandey, Michael A. Bender, and Rob Johnson
        let mask = 1 << rank;
        let one = unsafe{_pdep_u64(word, mask)};
        return one.trailing_zeros() as usize;
    }

    for _ in 0..rank {
        // reset the lowest set bits (BLSR)
        word &= word - 1;
    }
    word.trailing_zeros() as usize
}

#[inline(always)]
pub fn select_zero_in_word(word: u64, rank: usize) -> usize {
    select_in_word(!word, rank)
}