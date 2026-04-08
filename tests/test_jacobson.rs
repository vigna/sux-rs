use sux::bal_paren::jacobson::{JacobsonBP, JacobsonBPBitFieldVec};

/// Helper: build a balanced parentheses sequence from a pattern string
/// where '(' = open (1-bit) and ')' = close (0-bit).
fn from_pattern(pattern: &str) -> (Vec<u64>, usize) {
    let len = pattern.len();
    let mut words = vec![0u64; len.div_ceil(64)];
    for (i, c) in pattern.chars().enumerate() {
        if c == '(' {
            words[i / 64] |= 1u64 << (i % 64);
        }
    }
    (words, len)
}

/// Verifies that `find_close` is correct for all open parens in a pattern.
fn verify_find_close<O: value_traits::slices::SliceByValue<Value = usize>>(
    bp: &JacobsonBP<O>,
    pattern: &str,
) {
    let mut stack = Vec::new();
    let mut expected = vec![0usize; pattern.len()];
    for (i, c) in pattern.chars().enumerate() {
        if c == '(' {
            stack.push(i);
        } else {
            let open = stack.pop().unwrap();
            expected[open] = i;
        }
    }

    for (i, c) in pattern.chars().enumerate() {
        if c == '(' {
            assert_eq!(
                bp.find_close(i),
                Some(expected[i]),
                "find_close({i}) failed"
            );
        }
    }
}

// ── Basic tests ─────────────────────────────────────────────────────

#[test]
fn test_small_patterns() {
    for pattern in [
        "()",
        "(())",
        "()()",
        "(()())",
        "((()))",
        "(())(())",
        "((((()))))",
    ] {
        let (words, len) = from_pattern(pattern);
        let bp = JacobsonBP::new(words.clone(), len);
        verify_find_close(&bp, pattern);

        let bp_bfv = JacobsonBPBitFieldVec::new_with_bit_field_vec(words, len);
        verify_find_close(&bp_bfv, pattern);
    }
}

#[test]
fn test_cross_word_boundary() {
    // Build a sequence long enough to cross 64-bit word boundaries.
    // 40 nested pairs = 80 bits, forcing far matches.
    let n = 40;
    let pattern: String = "(".repeat(n) + &")".repeat(n);
    let (words, len) = from_pattern(&pattern);

    let bp = JacobsonBP::new(words.clone(), len);
    verify_find_close(&bp, &pattern);

    let bp_bfv = JacobsonBPBitFieldVec::new_with_bit_field_vec(words, len);
    verify_find_close(&bp_bfv, &pattern);
}

#[test]
fn test_large_random() {
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    let n = 1000; // number of pairs
    let mut rng = SmallRng::seed_from_u64(0);

    let total = 2 * n;
    let mut pattern = Vec::with_capacity(total);
    let mut opens = 0usize;
    let mut closes = 0usize;
    for i in 0..total {
        let remaining_opens = n - opens;
        let excess = opens - closes;
        let remaining = total - i;
        let must_close = excess == remaining;
        let can_open = remaining_opens > 0 && !must_close;
        let open = can_open && (excess == 0 || rng.random_bool(0.5));
        if open {
            pattern.push('(');
            opens += 1;
        } else {
            pattern.push(')');
            closes += 1;
        }
    }
    let pattern: String = pattern.into_iter().collect();

    let (words, len) = from_pattern(&pattern);

    let bp = JacobsonBP::new(words.clone(), len);
    verify_find_close(&bp, &pattern);

    let bp_bfv = JacobsonBPBitFieldVec::new_with_bit_field_vec(words, len);
    verify_find_close(&bp_bfv, &pattern);
}

// ── Both variants agree ─────────────────────────────────────────────

#[test]
fn test_comp_int_and_bfv_agree() {
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};
    use sux::bal_paren::jacobson::JacobsonBPPrefixSum;

    let n = 500;
    let mut rng = SmallRng::seed_from_u64(42);
    let mut bal = String::new();
    let mut depth = 0;
    for _ in 0..n {
        if depth == 0 || (depth < n && rng.random_bool(2.0 / 3.0)) {
            bal.push('(');
            depth += 1;
        } else {
            bal.push(')');
            depth -= 1;
        }
    }
    while depth > 0 {
        bal.push(')');
        depth -= 1;
    }

    let (words, len) = from_pattern(&bal);

    let bp_ci = JacobsonBP::new(words.clone(), len);
    let bp_bfv = JacobsonBPBitFieldVec::new_with_bit_field_vec(words.clone(), len);
    let bp_ps = JacobsonBPPrefixSum::new_with_prefix_sum(words, len);

    for i in 0..len {
        let ci = bp_ci.find_close(i);
        let bfv = bp_bfv.find_close(i);
        let ps = bp_ps.find_close(i);
        assert_eq!(ci, bfv, "CompInt vs BFV disagree at pos {i}");
        assert_eq!(ci, ps, "CompInt vs PrefixSum disagree at pos {i}");
    }
}

// ── Accessors ───────────────────────────────────────────────────────

#[test]
fn test_accessors() {
    let (words, len) = from_pattern("(())()");
    let bp = JacobsonBP::new(words, len);
    assert_eq!(bp.len(), 6);
    assert!(!bp.is_empty());
    assert_eq!(bp.words().len(), 1);
}

#[test]
fn test_empty() {
    let bp = JacobsonBP::new(vec![], 0);
    assert!(bp.is_empty());
    assert_eq!(bp.len(), 0);
    assert_eq!(bp.find_close(0), None);
}

// ── find_close edge cases ───────────────────────────────────────────

#[test]
fn test_find_close_not_open() {
    let bp = JacobsonBP::new(vec![0b0011], 4);
    assert_eq!(bp.find_close(2), None); // close paren
    assert_eq!(bp.find_close(4), None); // out of bounds
}

#[test]
fn test_find_close_sequential_pairs_across_words() {
    // 32 "()" pairs per word, two words — all matches are in-word.
    let bp = JacobsonBP::new(vec![0x5555_5555_5555_5555, 0x5555_5555_5555_5555], 128);
    for i in 0..64 {
        let pos = i * 2;
        assert_eq!(bp.find_close(pos), Some(pos + 1), "Failed for pos={pos}");
    }
}

// ── Serialization ───────────────────────────────────────────────────

#[cfg(feature = "epserde")]
#[test]
fn test_epserde() {
    use epserde::prelude::Aligned64;
    use epserde::utils::AlignedCursor;

    // Build a non-trivial BP sequence crossing word boundaries.
    let n = 200;
    let pattern: String = "(".repeat(n) + &")".repeat(n);
    let (words, len) = from_pattern(&pattern);
    let bp = JacobsonBP::new(words, len);

    // Serialize
    let mut cursor = <AlignedCursor<Aligned64>>::new();
    unsafe {
        use epserde::ser::Serialize;
        bp.serialize(&mut cursor).expect("Could not serialize");
    }

    let buf_len = cursor.len();
    cursor.set_position(0);

    // Deserialize
    let bp2 = unsafe {
        use epserde::deser::Deserialize;
        <JacobsonBP>::read_mem(&mut cursor, buf_len).expect("Could not deserialize")
    };
    let bp2 = bp2.uncase();

    assert_eq!(bp.len(), bp2.len());
    for i in 0..bp.len() {
        assert_eq!(
            bp.find_close(i),
            bp2.find_close(i),
            "epserde round-trip: find_close({i}) mismatch"
        );
    }
}
