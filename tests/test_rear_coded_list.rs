/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 * SPDX-FileCopyrightText: 2023 Tommaso Fontana
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use anyhow::Result;
use epserde::prelude::*;
use lender::*;
use rand::prelude::*;
use std::io::prelude::*;
use std::io::BufReader;
use sux::prelude::*;

#[test]
fn test_rear_coded_list_100() -> Result<()> {
    test_rear_coded_list("tests/data/wordlist.100")?;
    #[cfg(feature = "slow_tests")]
    test_rear_coded_list("tests/data/wordlist.10000")?;
    Ok(())
}

fn test_rear_coded_list(path: impl AsRef<str>) -> Result<()> {
    let words = BufReader::new(std::fs::File::open(path.as_ref()).unwrap())
        .lines()
        .map(|line| line.unwrap())
        .collect::<Vec<_>>();

    // test sorted RCL

    // create a new rca with u16 as pointers (this limit data to u16::MAX bytes max size)
    let mut rcab = <RearCodedListBuilder>::new(4);
    rcab.extend(words.iter().map(|s| s.as_str()).into_lender());

    rcab.print_stats();
    let rca = rcab.build();

    assert_eq!(rca.len(), words.len());

    // test that we can decode every string
    for (i, word) in words.iter().enumerate() {
        assert_eq!(&rca.get(i), word);
    }

    // test that the iter is correct
    for (i, word) in rca.iter().enumerate() {
        assert_eq!(word, words[i]);
    }

    for from in 0..rca.len() {
        for (i, word) in rca.iter_from(from).enumerate() {
            assert_eq!(word, words[i + from]);
        }
    }

    // test that the lend is correct
    for_![(i, word) in rca.lend().enumerate() {
        assert_eq!(word, words[i]);
    }];

    for from in 0..rca.len() {
        for_![(i, word) in rca.lend_from(from).enumerate() {
            assert_eq!(word, words[i + from]);
        }]
    }

    assert!(!rca.contains(""));

    for (i, word) in words.iter().enumerate() {
        assert!(rca.contains(word.as_str()));
        assert_eq!(rca.index_of(word.as_str()), Some(i));
        let mut word = word.clone();
        word.push_str("IT'S HIGHLY IMPROBABLE THAT THIS STRING IS IN THE WORDLIST");
        assert!(!rca.contains(word.as_str()));
        assert!(rca.index_of(word.as_str()).is_none());
    }

    let tmp_file = std::env::temp_dir().join("test_serdes_rcl.bin");
    let mut file = std::io::BufWriter::new(std::fs::File::create(&tmp_file)?);
    let schema = rca.serialize_with_schema(&mut file)?;
    drop(file);
    println!("{}", schema.to_csv());

    let c = <RearCodedList>::mmap(&tmp_file, epserde::deser::Flags::empty())?;

    for (i, word) in words.iter().enumerate() {
        assert_eq!(&c.get(i), word);
    }

    // test unsorted RCL

    let mut rcab = <RearCodedListBuilder>::new(4);
    let mut shuffled_words = words.iter().map(|s| s.as_str()).collect::<Vec<_>>();
    shuffled_words.shuffle(&mut rand::rng());

    for string in shuffled_words.iter() {
        rcab.push(string);
    }
    let rca = rcab.build();

    for (i, word) in shuffled_words.iter().enumerate() {
        assert_eq!(&rca.get(i), word);
    }

    let l = rca.len();
    let mut iter = rca.into_iter().enumerate();
    assert_eq!(iter.len(), l);
    while let Some((i, word)) = iter.next() {
        assert_eq!(word, shuffled_words[i]);
        assert_eq!(iter.len(), l - i - 1);
    }

    let mut iter = rca.into_lender().enumerate();
    assert_eq!(iter.len(), l);
    while let Some((i, word)) = iter.next() {
        assert_eq!(word, shuffled_words[i]);
        assert_eq!(iter.len(), l - i - 1);
    }

    for from in 0..rca.len() {
        for (i, word) in rca.iter_from(from).enumerate() {
            assert_eq!(word, shuffled_words[i + from]);
        }
    }

    assert!(!rca.contains(""));

    for (i, word) in shuffled_words.iter().enumerate() {
        assert!(rca.contains(*word));
        assert_eq!(rca.index_of(*word), Some(i));
        let mut word = word.to_string();
        word.push_str("IT'S HIGHLY IMPROBABLE THAT THIS STRING IS IN THE WORDLIST");
        assert!(!rca.contains(word.as_str()));
        assert!(rca.index_of(word.as_str()).is_none());
    }

    let tmp_file = std::env::temp_dir().join("test_serdes_rcl.bin");
    let mut file = std::io::BufWriter::new(std::fs::File::create(&tmp_file)?);
    let schema = rca.serialize_with_schema(&mut file)?;
    drop(file);
    println!("{}", schema.to_csv());

    let c = <RearCodedList>::mmap(&tmp_file, epserde::deser::Flags::empty())?;

    for (i, word) in shuffled_words.iter().enumerate() {
        assert_eq!(&c.get(i), word);
    }

    Ok(())
}
