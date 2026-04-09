use sux::bal_paren::jacobson::JacobsonBalParen;
use sux::bits::{BitFieldVec, BitVec};
use sux::list::prefix_sum_int_list::PrefixSumIntList;
use sux::prelude::BalParen;
use sux::traits::{BitLength, PredUnchecked};
use value_traits::slices::SliceByValue;

/// Helper: build a balanced parentheses sequence from a pattern string
/// where '(' = open (1-bit) and ')' = close (0-bit).
fn from_pattern(pattern: &str) -> BitVec<Box<[usize]>> {
    let len = pattern.len();
    let mut words = vec![0usize; len.div_ceil(usize::BITS as usize)];
    for (i, c) in pattern.chars().enumerate() {
        if c == '(' {
            words[i / usize::BITS as usize] |= 1usize << (i % usize::BITS as usize);
        }
    }
    unsafe { BitVec::from_raw_parts(words.into_boxed_slice(), len) }
}

/// Verifies that `find_close` is correct for all open parens in a pattern.
fn verify_find_close<
    B: AsRef<[usize]> + BitLength,
    P: for<'a> PredUnchecked<Input = usize, Output<'a> = usize>,
    O: SliceByValue<Value = usize>,
>(
    bp: &JacobsonBalParen<B, P, O>,
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
        let paren = from_pattern(pattern);
        let bp = JacobsonBalParen::new(&paren);
        verify_find_close(&bp, pattern);

        let bp_bfv =
            <JacobsonBalParen<_, _, BitFieldVec<Box<[usize]>>>>::new_with_bit_field_vec(paren);
        verify_find_close(&bp_bfv, pattern);
    }
}

#[test]
fn test_cross_word_boundary() {
    // Build a sequence long enough to cross 64-bit word boundaries.
    // 40 nested pairs = 80 bits, forcing far matches.
    let n = 40;
    let pattern: String = "(".repeat(n) + &")".repeat(n);
    let paren = from_pattern(&pattern);

    let bp = JacobsonBalParen::new(&paren);
    verify_find_close(&bp, &pattern);

    let bp_bfv =
        <JacobsonBalParen<_, _, BitFieldVec<Box<[usize]>>>>::new_with_bit_field_vec(&paren);
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

    let paren = from_pattern(&pattern);

    let bp = JacobsonBalParen::new(&paren);
    verify_find_close(&bp, &pattern);

    let bp_bfv = <JacobsonBalParen<_, _, BitFieldVec<Box<[usize]>>>>::new_with_bit_field_vec(paren);
    verify_find_close(&bp_bfv, &pattern);
}

// ── Both variants agree ─────────────────────────────────────────────

#[test]
fn test_all_offset_variants_agree() {
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

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

    let paren = from_pattern(&bal);

    let bp_ci = JacobsonBalParen::new(&paren);
    let bp_bfv =
        <JacobsonBalParen<_, _, BitFieldVec<Box<[usize]>>>>::new_with_bit_field_vec(&paren);
    let bp_ps = <JacobsonBalParen<_, _, PrefixSumIntList>>::new_with_prefix_sum(&paren);

    for i in 0..paren.len() {
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
    let paren = from_pattern("(())()");
    let bp = JacobsonBalParen::new(paren);
    assert_eq!(bp.len(), 6);
    assert_eq!(bp.as_ref().len(), 1);
}

#[test]
fn test_empty() {
    let bp = JacobsonBalParen::new(BitVec::new(0));
    assert_eq!(bp.len(), 0);
}

#[test]
#[should_panic(expected = "out of bounds")]
fn test_find_close_empty_panics() {
    let bp = JacobsonBalParen::new(BitVec::new(0));
    bp.find_close(0);
}

// ── find_close edge cases ───────────────────────────────────────────

#[test]
fn test_find_close_not_open() {
    let bp = JacobsonBalParen::new(unsafe { BitVec::from_raw_parts(vec![0b0011], 4) });
    assert_eq!(bp.find_close(2), None); // close paren
}

#[test]
#[should_panic(expected = "out of bounds")]
fn test_find_close_out_of_bounds() {
    let bp = JacobsonBalParen::new(unsafe { BitVec::from_raw_parts(vec![0b0011], 4) });
    bp.find_close(4);
}

#[test]
fn test_find_close_sequential_pairs_across_words() {
    // Pattern 0b...0101_0101: each "()" pair is in-word.
    let alternating: usize = usize::MAX / 3; // 0x5555...5555
    let bp = JacobsonBalParen::new(unsafe {
        BitVec::from_raw_parts(vec![alternating; 2], 2 * usize::BITS as usize)
    });
    for i in 0..usize::BITS as usize {
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
    let paren = from_pattern(&pattern);
    let bp = JacobsonBalParen::new(&paren);

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
        <JacobsonBalParen>::read_mem(&mut cursor, buf_len).expect("Could not deserialize")
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
