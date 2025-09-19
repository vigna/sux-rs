/*
 * SPDX-FileCopyrightText: 2025 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

#[test]
fn fail() {
    // Tests must be specifying one-by-one as we need to feature gate them.
    let _t = trybuild::TestCases::new();
    #[cfg(feature = "epserde")]
    _t.compile_fail("tests/fail/drop_memcase_and_get.rs");
}
