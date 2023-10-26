/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 * SPDX-FileCopyrightText: 2023 Tommaso Fontana
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use anyhow::Result;
use epserde::prelude::*;
use std::io::prelude::*;
use std::io::BufReader;
use sux::prelude::*;

#[test]
fn test_negative_redundancy() {
    let mut rclb = RearCodedListBuilder::new(10000);
    rclb.push("UkVBRE1FLm1k");
    rclb.push("VE9ETy50eHQ=");
    rclb.push("b2xk");
    rclb.push("b2xkcHJvamVjdA==");
    rclb.push("cGFyc2VyLmM=");
    rclb.push("cmVmcy90YWdzL3YxLjA=");
    rclb.push("cmVmcy90YWdzL3YyLjAtYW5vbnltb3Vz");
    rclb.push("cmVmcy9oZWFkcy9tYXN0ZXI=");
    rclb.push("dGVzdHM=");

    rclb.print_stats();
}

#[test]
fn test_rear_coded_list() -> Result<()> {
    let words = BufReader::new(std::fs::File::open("tests/data/wordlist.10000").unwrap())
        .lines()
        .map(|line| line.unwrap())
        .collect::<Vec<_>>();

    // create a new rca with u16 as pointers (this limit data to u16::MAX bytes max size)
    let mut rcab = <RearCodedListBuilder>::new(8);
    rcab.extend(words.iter());

    rcab.print_stats();
    let rca = rcab.build();

    assert_eq!(rca.len(), words.len());

    // test that we can decode every string
    for (i, word) in words.iter().enumerate() {
        assert_eq!(&rca.get(i), word);
    }

    // test that the iter is correct
    for (i, word) in rca.into_iter().enumerate() {
        assert_eq!(word, words[i]);
    }

    for from in 0..rca.len() {
        for (i, word) in rca.into_iter_from(from).enumerate() {
            assert_eq!(word, words[i + from]);
        }
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

    let c = <RearCodedList>::mmap(&tmp_file, epserde::deser::Flags::empty())?;

    for (i, word) in words.iter().enumerate() {
        assert_eq!(&c.get(i), word);
    }

    Ok(())
}
