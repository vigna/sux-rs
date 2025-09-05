/*
 * SPDX-FileCopyrightText: 2025 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

#[test]
fn fail() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/fail/*.rs");
}
