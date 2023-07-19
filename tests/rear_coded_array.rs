use std::io::prelude::*;
use std::io::BufReader;
use sux::prelude::*;

#[test]
fn test_rear_coded_array() {
    let words = BufReader::new(std::fs::File::open("tests/data/wordlist.10000").unwrap())
        .lines()
        .map(|line| line.unwrap())
        .collect::<Vec<_>>();

    // create a new rca with u16 as pointers (this limit data to u16::MAX bytes max size)
    let mut rca = <RearCodedArray<u16>>::new(8);
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
}
