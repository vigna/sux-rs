/*
 * SPDX-FileCopyrightText: 2025 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use epserde::{Epserde, deser::Deserialize, ser::Serialize};
use std::io::Cursor;
use sux::traits::{IndexedSeq, Types};

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "epserde", derive(epserde::Epserde))]
struct Wrapper<A>(A);

// A newtype for Vec<String> returning an &str on a get
impl<S: AsRef<str>> Types for Wrapper<Vec<S>> {
    type Input = String;
    type Output<'a> = &'a str;
}

impl<S: AsRef<str>> IndexedSeq for Wrapper<Vec<S>> {
    unsafe fn get_unchecked(&self, index: usize) -> Self::Output<'_> {
        unsafe { self.0.get_unchecked(index).as_ref() }
    }

    fn len(&self) -> usize {
        self.0.len()
    }
}

fn main() {
    let vec = Wrapper(vec![
        "foo".to_string(),
        "bar".to_string(),
        "baz".to_string(),
    ]);

    let mut buffer = Vec::new();
    unsafe { vec.serialize(&mut buffer).unwrap() };
    let cursor = Cursor::new(&buffer);
    let mem_case = unsafe { <Wrapper<Vec<String>>>::read_mem(cursor, buffer.len()).unwrap() };

    let s = mem_case.get(0);
    drop(mem_case);
    assert_eq!(s, "foo");
}
