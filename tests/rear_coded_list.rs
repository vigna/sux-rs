/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use anyhow::Result;
use epserde::*;
use std::io::prelude::*;
use std::io::BufReader;
use sux::prelude::*;

#[test]
fn test_rear_coded_list() -> Result<()> {
    let words = BufReader::new(std::fs::File::open("tests/data/wordlist.10000").unwrap())
        .lines()
        .map(|line| line.unwrap())
        .collect::<Vec<_>>();

    // create a new rca with u16 as pointers (this limit data to u16::MAX bytes max size)
    let mut rca = <RearCodedList<u16>>::new(8);
    rca.extend(words.iter());

    rca.print_stats();

    assert_eq!(rca.len(), words.len());

    // test that we can decode every string
    for (i, word) in words.iter().enumerate() {
        assert_eq!(&rca.get(i), word);
    }

    // test that the iter is correct
    for (i, word) in rca.iter().enumerate() {
        assert_eq!(word, words[i]);
    }

    assert!(!rca.contains(""));

    for word in words.iter() {
        assert!(rca.contains(word));
        let mut word = word.clone();
        word.push_str("IT'S HIGHLY IMPROBABLE THAT THIS STRING IS IN THE WORDLIST");
        assert!(!rca.contains(&word));
    }

    let tmp_file = std::env::temp_dir().join("test_serdes_rcl.bin");
    let mut file = std::io::BufWriter::new(std::fs::File::create(&tmp_file)?);
    let schema = rca.serialize_with_schema(&mut file)?;
    drop(file);
    println!("{}", schema.to_csv());

    let c = <RearCodedList<u16>>::mmap(&tmp_file, epserde::Flags::empty())?;

    for (i, word) in words.iter().enumerate() {
        assert_eq!(&c.get(i), word);
    }

    Ok(())
}
