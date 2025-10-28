/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 * SPDX-FileCopyrightText: 2023 Tommaso Fontana
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

#[cfg(test)]
use lender::{IntoLender, Lending};
use sux::{dict::RearCodedListBuilder, traits::IndexedSeq};

#[cfg(feature = "epserde")]
mod test {
    use anyhow::Result;
    use epserde::deser::Deserialize;
    use epserde::ser::Serialize;
    use epserde::utils::AlignedCursor;
    use indexed_dict::*;
    use lender::*;
    use rand::prelude::*;
    use std::io::{BufRead, BufReader};
    use sux::prelude::*;

    #[test]
    fn test_rear_coded_list_100() -> Result<()> {
        test_rear_coded_list("tests/data/wordlist.100")?;
        #[cfg(feature = "slow_tests")]
        test_rear_coded_list("tests/data/wordlist.10000")?;
        Ok(())
    }

    fn test_rear_coded_list(path: impl AsRef<str>) -> Result<()> {
        use maligned::A16;
        let words = BufReader::new(std::fs::File::open(path.as_ref())?)
            .lines()
            .collect::<Result<Vec<_>, _>>()?;

        // test sorted RCL

        // create a new rca with u16 as pointers (this limit data to u16::MAX bytes max size)
        let mut rcab = <RearCodedListBuilder<str, true>>::new(4);
        rcab.extend(words.iter().into_lender());

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
        for_![(i, word) in rca.lender().enumerate() {
            assert_eq!(word, words[i]);
        }];

        for from in 0..rca.len() {
            for_![(i, word) in rca.lender_from(from).enumerate() {
                assert_eq!(word, words[i + from]);
            }]
        }

        assert!(!rca.contains(""));

        for (i, word) in words.iter().enumerate() {
            assert!(rca.contains(word.as_str()));
            assert_eq!(rca.index_of(word.as_str()), Some(i));
            let mut word = word.clone();
            word.push_str("IT'S HIGHLY IMPROBABLE THAT THIS STRING IS IN THE WORD LIST");
            assert!(!rca.contains(word.as_str()));
            assert!(rca.index_of(word.as_str()).is_none());
        }

        let mut cursor = <AlignedCursor<A16>>::new();
        let schema = unsafe { rca.serialize_with_schema(&mut cursor)? };
        println!("{}", schema.to_csv());

        let len = cursor.len();
        cursor.set_position(0);
        let c = unsafe {
            <RearCodedListStr<true>>::read_mmap(&mut cursor, len, epserde::deser::Flags::empty())?
        };
        let c = c.uncase();

        for (i, word) in words.iter().enumerate() {
            assert_eq!(&c.get(i), word);
        }

        // test unsorted RCL

        let mut rcl_builder = <RearCodedListBuilder<str, false>>::new(4);
        let mut shuffled_words = words.iter().map(|s| s.as_str()).collect::<Vec<_>>();
        shuffled_words.shuffle(&mut rand::rng());

        for string in shuffled_words.iter() {
            rcl_builder.push(*string);
        }
        let rca = rcl_builder.build();

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

        // Note: unsorted RCL does not support contains/index_of (IndexedDict trait is only for SORTED=true)

        let mut cursor = <AlignedCursor<A16>>::new();
        let schema = unsafe { rca.serialize_with_schema(&mut cursor)? };
        println!("{}", schema.to_csv());

        let len = cursor.len();
        cursor.set_position(0);
        let c = unsafe {
            <RearCodedListStr<false>>::read_mmap(&mut cursor, len, epserde::deser::Flags::empty())?
        };
        let c = c.uncase();

        for (i, word) in shuffled_words.iter().enumerate() {
            assert_eq!(&c.get(i), word);
        }

        Ok(())
    }

    #[test]
    #[should_panic(expected = "Strings must be sorted in ascending order")]
    fn test_panics_on_out_of_order() {
        let mut rcab = <RearCodedListBuilder<str, true>>::new(4);
        rcab.push("apple");
        rcab.push("banana");
        rcab.push("cherry");
        // This should panic because "apricot" < "cherry"
        rcab.push("apricot");
    }
}

#[cfg(test)]
fn read_into_lender<L: IntoLender>(into_lender: L) -> usize
where
    for<'a> <L::Lender as Lending<'a>>::Lend: AsRef<str>,
{
    use lender::Lender;

    let mut iter = into_lender.into_lender();
    let mut c = 0;
    while let Some(s) = iter.next() {
        c += s.as_ref().len();
    }

    c
}

#[test]
fn test_into_lend() {
    let mut builder = RearCodedListBuilder::<str, true>::new(4);
    builder.push("a");
    builder.push("b");
    builder.push("c");
    builder.push("d");
    let rcl = builder.build();
    read_into_lender(&rcl);
}

#[test]
fn test_zero_bytes() {
    let strings = vec![
        "\0\0\0\0a",
        "\0\0\0b",
        "\0\0c",
        "\0d",
        "e",
        "f\0",
        "g\0\0",
        "h\0\0\0",
    ];
    let mut builder = RearCodedListBuilder::<str, true>::new(4);
    for &s in &strings {
        builder.push(s);
    }
    let rcl = builder.build();
    for i in 0..rcl.len() {
        let s = rcl.get(i);
        assert_eq!(s, strings[i]);
    }
}

#[cfg(feature = "epserde")]
#[test]
fn test_ser_str() -> anyhow::Result<()> {
    use epserde::utils::AlignedCursor;
    use sux::traits::{IndexedDict, IndexedSeq};
    use sux::{dict::rear_coded_list::serialize_str, utils::FromIntoIterator};

    let v = ["a", "ab", "ab", "abc", "b", "bb"];

    let mut cursor = AlignedCursor::<maligned::A16>::new();
    serialize_str::<_, _, true>(4, FromIntoIterator::from(v.clone()), &mut cursor)?;

    cursor.set_position(0);
    let deser = unsafe {
        use epserde::deser::Deserialize;
        use sux::dict::RearCodedListStr;
        RearCodedListStr::<true>::deserialize_full(&mut cursor)?
    };
    assert_eq!(deser.len(), 6);
    for (i, s) in deser.iter().enumerate() {
        assert_eq!(s, v[i]);
    }
    assert_eq!(deser.get(0), "a");
    assert_eq!(deser.get(1), "ab");
    assert_eq!(deser.get(2), "ab");
    assert_eq!(deser.get(3), "abc");
    assert_eq!(deser.get(4), "b");
    assert_eq!(deser.get(5), "bb");
    assert_eq!(deser.index_of("a"), Some(0));
    assert_eq!(deser.index_of("ab"), Some(1));
    assert_eq!(deser.index_of("abc"), Some(3));
    assert_eq!(deser.index_of("b"), Some(4));
    assert_eq!(deser.index_of("bb"), Some(5));
    assert_eq!(deser.index_of("c"), None);

    let mut buf = String::new();
    deser.get_in_place(0, &mut buf);
    assert_eq!(&buf, "a");
    deser.get_in_place(1, &mut buf);
    assert_eq!(&buf, "ab");
    deser.get_in_place(2, &mut buf);
    assert_eq!(&buf, "ab");
    deser.get_in_place(3, &mut buf);
    assert_eq!(&buf, "abc");
    deser.get_in_place(4, &mut buf);
    assert_eq!(&buf, "b");
    deser.get_in_place(5, &mut buf);
    assert_eq!(&buf, "bb");

    Ok(())
}

#[cfg(feature = "epserde")]
#[test]
fn test_ser_slice() -> anyhow::Result<()> {
    use epserde::{deser::Deserialize, utils::AlignedCursor};
    use sux::dict::rear_coded_list::serialize_slice_u8;
    use sux::traits::{IndexedDict, IndexedSeq};
    use sux::utils::FromIntoIterator;

    let v = vec![
        vec![1u8],
        vec![1u8, 2u8],
        vec![1u8, 2u8],
        vec![1u8, 2u8, 3u8],
        vec![2u8],
        vec![2u8, 2u8],
    ];

    let mut cursor = AlignedCursor::<maligned::A16>::new();
    serialize_slice_u8::<_, _, true>(4, FromIntoIterator::from(v.clone()), &mut cursor)?;

    cursor.set_position(0);
    let deser = unsafe {
        use sux::dict::RearCodedListSliceU8;
        RearCodedListSliceU8::<true>::deserialize_full(&mut cursor)?
    };
    assert_eq!(deser.len(), 6);
    deser.iter().zip(v.iter()).for_each(|(s, t)| {
        assert_eq!(&s, t);
    });
    assert_eq!(deser.get(0), &[1u8]);
    assert_eq!(deser.get(1), &[1u8, 2u8]);
    assert_eq!(deser.get(2), &[1u8, 2u8]);
    assert_eq!(deser.get(3), &[1u8, 2u8, 3u8]);
    assert_eq!(deser.get(4), &[2u8]);
    assert_eq!(deser.get(5), &[2u8, 2u8]);
    assert_eq!(deser.index_of(vec![1u8]), Some(0));
    assert_eq!(deser.index_of(vec![1u8, 2u8]), Some(1));
    assert_eq!(deser.index_of(vec![1u8, 2u8, 3u8]), Some(3));
    assert_eq!(deser.index_of(vec![2u8]), Some(4));
    assert_eq!(deser.index_of(vec![2u8, 2u8]), Some(5));
    assert_eq!(deser.index_of(vec![3u8]), None);

    let mut buf = vec![];
    deser.get_in_place(0, &mut buf);
    assert_eq!(&buf, &[1u8]);
    deser.get_in_place(1, &mut buf);
    assert_eq!(&buf, &[1u8, 2u8]);
    deser.get_in_place(2, &mut buf);
    assert_eq!(&buf, &[1u8, 2u8]);
    deser.get_in_place(3, &mut buf);
    assert_eq!(&buf, &[1u8, 2u8, 3u8]);
    deser.get_in_place(4, &mut buf);
    assert_eq!(&buf, &[2u8]);
    deser.get_in_place(5, &mut buf);
    assert_eq!(&buf, &[2u8, 2u8]);
    Ok(())
}
