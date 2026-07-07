/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use sux::prelude::*;

const LEN: usize = 130;
const WORD_BITS: usize = 64;
const LOGICAL_ONES: [usize; 13] = [0, 1, 5, 31, 32, 63, 64, 65, 70, 95, 126, 127, 129];
const RANK_PROBES: [usize; 12] = [0, 1, 2, 6, 63, 64, 65, 66, 96, 127, 128, LEN];

fn clean_words() -> Vec<u64> {
    let mut words = vec![0; LEN.div_ceil(WORD_BITS)];
    for &pos in &LOGICAL_ONES {
        words[pos / WORD_BITS] |= 1u64 << (pos % WORD_BITS);
    }
    words
}

fn dirty_words() -> Vec<u64> {
    let mut words = clean_words();
    let residual = LEN % WORD_BITS;
    assert_ne!(residual, 0);
    words[LEN / WORD_BITS] |= !0u64 << residual;
    words
}

fn clean_bit_vec() -> BitVec<Vec<u64>> {
    // SAFETY: `clean_words` returns exactly `LEN.div_ceil(WORD_BITS)` initialized
    // 64-bit words, so `LEN` bits fit in the owned backing storage.
    unsafe { BitVec::from_raw_parts(clean_words(), LEN) }
}

fn dirty_bit_vec() -> BitVec<Vec<u64>> {
    // SAFETY: `dirty_words` preserves the same owned backing length as
    // `clean_words`; only padding bits after `LEN` are set.
    unsafe { BitVec::from_raw_parts(dirty_words(), LEN) }
}

fn expected_rank(pos: usize) -> usize {
    LOGICAL_ONES.iter().filter(|&&one| one < pos).count()
}

fn assert_rank_answers<R: Rank>(clean: &R, dirty: &R) {
    assert_eq!(clean.num_ones(), LOGICAL_ONES.len());
    assert_eq!(dirty.num_ones(), LOGICAL_ONES.len());

    for pos in RANK_PROBES {
        let expected = expected_rank(pos);
        assert_eq!(clean.rank(pos), expected, "clean rank({pos})");
        assert_eq!(dirty.rank(pos), clean.rank(pos), "dirty rank({pos})");
        assert_eq!(dirty.rank(pos), expected, "dirty rank({pos})");
    }

    assert_eq!(clean.rank(LEN), LOGICAL_ONES.len());
    assert_eq!(dirty.rank(LEN), LOGICAL_ONES.len());
}

fn assert_select_answers<S: Select>(clean: &S, dirty: &S) {
    assert_eq!(clean.num_ones(), LOGICAL_ONES.len());
    assert_eq!(dirty.num_ones(), LOGICAL_ONES.len());

    for (rank, &expected) in LOGICAL_ONES.iter().enumerate() {
        let clean_selected = clean.select(rank);
        let dirty_selected = dirty.select(rank);
        let dirty_pos = dirty_selected.unwrap_or(usize::MAX);

        assert_eq!(clean_selected, Some(expected), "clean select({rank})");
        assert!(
            dirty_pos < LEN,
            "dirty select({rank}) returned padding position {dirty_pos}"
        );
        assert_eq!(dirty_selected, clean_selected, "dirty select({rank})");
        assert_eq!(dirty_selected, Some(expected), "dirty select({rank})");
    }

    assert_eq!(clean.select(LOGICAL_ONES.len()), None);
    assert_eq!(dirty.select(LOGICAL_ONES.len()), None);
}

#[test]
fn test_rank9_ignores_dirty_padding() {
    let clean = Rank9::new(clean_bit_vec());
    let dirty = Rank9::new(dirty_bit_vec());

    assert_rank_answers(&clean, &dirty);
}

#[test]
fn test_rank_small_ignores_dirty_padding() {
    let clean = rank_small![u64: 0; clean_bit_vec()];
    let dirty = rank_small![u64: 0; dirty_bit_vec()];

    assert_rank_answers(&clean, &dirty);
}

#[test]
fn test_select9_ignores_dirty_padding() {
    let clean = Select9::new(Rank9::new(clean_bit_vec()));
    let dirty = Select9::new(Rank9::new(dirty_bit_vec()));

    assert_rank_answers(&clean, &dirty);
    assert_select_answers(&clean, &dirty);
}

#[test]
fn test_select_adapt_ignores_dirty_padding() {
    let clean_bits: AddNumBits<_> = clean_bit_vec().into();
    let dirty_bits: AddNumBits<_> = dirty_bit_vec().into();
    let clean = SelectAdapt::new(clean_bits);
    let dirty = SelectAdapt::new(dirty_bits);

    assert_select_answers(&clean, &dirty);
}

#[test]
fn test_select_adapt_const_ignores_dirty_padding() {
    let clean_bits: AddNumBits<_> = clean_bit_vec().into();
    let dirty_bits: AddNumBits<_> = dirty_bit_vec().into();
    let clean = SelectAdaptConst::<_, _>::new(clean_bits);
    let dirty = SelectAdaptConst::<_, _>::new(dirty_bits);

    assert_select_answers(&clean, &dirty);
}

#[test]
fn test_select_small_ignores_dirty_padding() {
    let clean = SelectSmall::<2, 9, _>::new(RankSmall::<64, 2, 9, _>::new(clean_bit_vec()));
    let dirty = SelectSmall::<2, 9, _>::new(RankSmall::<64, 2, 9, _>::new(dirty_bit_vec()));

    assert_rank_answers(&clean, &dirty);
    assert_select_answers(&clean, &dirty);
}
