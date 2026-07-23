/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

use sux::prelude::*;
use sux::traits::BitVecOps;

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
    // SAFETY: clean_words returns exactly LEN.div_ceil(WORD_BITS) initialized
    // 64-bit words, so LEN bits fit in the owned backing storage.
    unsafe { BitVec::from_raw_parts(clean_words(), LEN) }
}

fn dirty_bit_vec() -> BitVec<Vec<u64>> {
    // SAFETY: dirty_words preserves the same owned backing length as
    // clean_words; only padding bits after LEN are set.
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

/// Builds a BitVec via safe `push`, then `pop`s down so the backing retains
/// stale one-bits in words beyond `len().div_ceil(word_bits)`. Bits
/// `0..keep` are all ones, so `select(r) == Some(r)` and `rank(keep) == keep`.
fn popped_ones(fill: usize, keep: usize) -> BitVec<Vec<u64>> {
    let mut b = BitVec::<Vec<u64>>::new(0);
    for _ in 0..fill {
        b.push(true);
    }
    for _ in 0..fill - keep {
        b.pop();
    }
    assert_eq!(b.len(), keep);
    assert_eq!(b.count_ones(), keep);
    b
}

fn assert_stale_select<S: Select>(s: &S, keep: usize) {
    assert_eq!(s.num_ones(), keep);
    for r in 0..keep {
        assert_eq!(s.select(r), Some(r), "select({r})");
    }
    assert_eq!(s.select(keep), None, "select past the last one");
}

fn assert_stale_rank<R: Rank>(r: &R, keep: usize) {
    assert_eq!(r.num_ones(), keep);
    assert_eq!(r.rank(keep), keep);
    assert_eq!(r.rank(keep / 2), keep / 2);
}

#[test]
fn test_rank9_ignores_stale_backing_words() {
    assert_stale_rank(&Rank9::new(popped_ones(130, 40)), 40);
}

#[test]
fn test_rank_small_ignores_stale_backing_words() {
    assert_stale_rank(&RankSmall::<64, 2, 9, _>::new(popped_ones(130, 40)), 40);
}

#[test]
fn test_select9_ignores_stale_backing_words() {
    // 600 logical ones cross a 512-quantum boundary while the stale backing
    // holds 1200, so a builder scanning stale words corrupts the inventory.
    let s = Select9::new(Rank9::new(popped_ones(1200, 600)));
    assert_stale_rank(&s, 600);
    assert_stale_select(&s, 600);
}

#[test]
fn test_select_adapt_ignores_stale_backing_words() {
    let bits: AddNumBits<_> = popped_ones(130, 40).into();
    assert_stale_select(&SelectAdapt::new(bits), 40);
}

#[test]
fn test_select_adapt_const_ignores_stale_backing_words() {
    let bits: AddNumBits<_> = popped_ones(130, 40).into();
    assert_stale_select(&SelectAdaptConst::<_, _>::new(bits), 40);
}

#[test]
fn test_select_small_ignores_stale_backing_words() {
    let s = SelectSmall::<2, 9, _>::new(RankSmall::<64, 2, 9, _>::new(popped_ones(1200, 600)));
    assert_stale_rank(&s, 600);
    assert_stale_select(&s, 600);
}

#[test]
fn test_full_word_boundary_ignores_stale_backing_words() {
    // len % 64 == 0: the tail mask is a no-op, so only the logical word
    // bound protects against the stale third backing word.
    assert_stale_rank(&Rank9::new(popped_ones(192, 128)), 128);

    let s = Select9::new(Rank9::new(popped_ones(192, 128)));
    assert_stale_rank(&s, 128);
    assert_stale_select(&s, 128);

    let bits: AddNumBits<_> = popped_ones(192, 128).into();
    assert_stale_select(&SelectAdapt::new(bits), 128);
}
