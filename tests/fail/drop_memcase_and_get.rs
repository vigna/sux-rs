/*
 * SPDX-FileCopyrightText: 2025 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use epserde::{deser::MemCase, Epserde};
use sux::traits::{IndexedSeq, Types};

#[derive(Epserde, Debug, Clone, PartialEq, Eq)]
struct VecString(Vec<String>);

impl Types for VecString {
    type Input = String;
    type Output<'a> = &'a str;
}

impl IndexedSeq for VecString {
    unsafe fn get_unchecked(&self, index: usize) -> Self::Output<'_> {
        self.0.get_unchecked(index).as_str()
    }

    fn len(&self) -> usize {
        self.0.len()
    }
}

fn main() {
    let vec = VecString(vec![
        "foo".to_string(),
        "bar".to_string(),
        "baz".to_string(),
    ]);
    let mem_case = MemCase::<VecString>::encase(vec);
    let s = mem_case.get(0);
    drop(mem_case);
    assert_eq!(s, "foo");
}
