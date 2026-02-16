/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 * SPDX-FileCopyrightText: 2023 Tommaso Fontana
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use sux::{
    bits::{BitFieldVec, bit_field_vec},
    dict::{RearCodedListBuilder, mapped_rear_coded_list::MappedRearCodedList},
    traits::IndexedSeq,
};

#[cfg(feature = "epserde")]
mod tests {
    use anyhow::Result;
    use epserde::deser::Deserialize;
    use epserde::ser::Serialize;
    use epserde::utils::AlignedCursor;
    use indexed_dict::*;
    use lender::*;
    use std::io::{BufRead, BufReader};
    use sux::{dict::mapped_rear_coded_list::MappedRearCodedList, prelude::*};

    #[test]
    fn test_perm_rear_coded_list_100() -> Result<()> {
        test_perm_rear_coded_list("tests/data/wordlist.100")?;
        #[cfg(feature = "slow_tests")]
        test_perm_rear_coded_list("tests/data/wordlist.10000")?;
        Ok(())
    }

    fn test_perm_rear_coded_list(path: impl AsRef<str>) -> Result<()> {
        use epserde::prelude::Aligned16;
        let words = BufReader::new(std::fs::File::open(path.as_ref())?)
            .lines()
            .collect::<Result<Vec<_>, _>>()?;
        let len = words.len();

        // Sorted

        let mut rclb = <RearCodedListBuilder<str, true>>::new(4);
        rclb.extend(words.iter().into_lender());

        rclb.print_stats();
        let rcl = rclb.build();

        assert_eq!(rcl.len(), len);

        let perm = (0..rcl.len()).rev().collect::<Vec<_>>().into_boxed_slice();
        let mrcl = MappedRearCodedList::from_parts(rcl, perm);
        assert_eq!(mrcl.len(), len);

        // test that we can decode every string
        for (i, word) in words.iter().enumerate() {
            assert_eq!(&mrcl.get(len - 1 - i), word);
        }

        // test that the iter is correct
        for (i, word) in mrcl.iter().enumerate() {
            dbg!(i, &word, &words[len - 1 - i]);
            assert_eq!(word, words[len - 1 - i]);
        }

        for from in 0..mrcl.len() {
            for (i, word) in mrcl.iter_from(from).enumerate() {
                assert_eq!(word, words[len - 1 - (i + from)]);
            }
        }

        // test that the lend is correct
        for_![(i, word) in mrcl.lender().enumerate() {
            assert_eq!(word, words[len - 1 - i]);
        }];

        for from in 0..mrcl.len() {
            for_![(i, word) in mrcl.lender_from(from).enumerate() {
                assert_eq!(word, words[len - 1 - (i + from)]);
            }]
        }

        let mut cursor = <AlignedCursor<Aligned16>>::new();
        let schema = unsafe { mrcl.serialize_with_schema(&mut cursor)? };
        println!("{}", schema.to_csv());

        let cursor_len = cursor.len();
        cursor.set_position(0);
        let c = unsafe {
            <MappedRearCodedList<str, String, Box<[u8]>, Box<[usize]>, Box<[usize]>, true>>::read_mmap(
                &mut cursor,
                cursor_len,
                epserde::deser::Flags::empty(),
            )?
        };
        let c = c.uncase();
        assert_eq!(c.len(), len);

        for (i, word) in words.iter().enumerate() {
            assert_eq!(&c.get(len - 1 - i), word);
        }

        // Unsorted

        let mut rclb = <RearCodedListBuilder<str, false>>::new(4);
        rclb.extend(words.iter().into_lender());

        rclb.print_stats();
        let rcl = rclb.build();

        assert_eq!(rcl.len(), len);

        let perm = (0..rcl.len()).rev().collect::<Vec<_>>().into_boxed_slice();
        let mrcl = MappedRearCodedList::from_parts(rcl, perm);
        assert_eq!(mrcl.len(), len);

        // test that we can decode every string
        for (i, word) in words.iter().enumerate() {
            assert_eq!(&mrcl.get(len - 1 - i), word);
        }

        // test that the iter is correct
        for (i, word) in mrcl.iter().enumerate() {
            dbg!(i, &word, &words[len - 1 - i]);
            assert_eq!(word, words[len - 1 - i]);
        }

        for from in 0..mrcl.len() {
            for (i, word) in mrcl.iter_from(from).enumerate() {
                assert_eq!(word, words[len - 1 - (i + from)]);
            }
        }

        // test that the lend is correct
        for_![(i, word) in mrcl.lender().enumerate() {
            assert_eq!(word, words[len - 1 - i]);
        }];

        for from in 0..mrcl.len() {
            for_![(i, word) in mrcl.lender_from(from).enumerate() {
                assert_eq!(word, words[len - 1 - (i + from)]);
            }]
        }

        let mut cursor = <AlignedCursor<Aligned16>>::new();
        let schema = unsafe { mrcl.serialize_with_schema(&mut cursor)? };
        println!("{}", schema.to_csv());

        let cursor_len = cursor.len();
        cursor.set_position(0);
        let c = unsafe {
            <MappedRearCodedList<str, String, Box<[u8]>, Box<[usize]>, Box<[usize]>, false>>::read_mmap(
                &mut cursor,
                cursor_len,
                epserde::deser::Flags::empty(),
            )?
        };
        let c = c.uncase();
        assert_eq!(c.len(), len);

        for (i, word) in words.iter().enumerate() {
            assert_eq!(&c.get(len - 1 - i), word);
        }

        Ok(())
    }
}

#[test]
fn test_bit_field_vec_mapped_rear_coded_list() {
    let mut builder = RearCodedListBuilder::<str, true>::new(4);
    builder.push("a");
    builder.push("b");
    builder.push("c");
    builder.push("d");
    let mrcl = <MappedRearCodedList<
        str,
        String,
        Box<[u8]>,
        Box<[usize]>,
        BitFieldVec<usize>,
        true,
    >>::from_parts(builder.build(), bit_field_vec![2; 3, 2, 1, 0]);

    assert_eq!(mrcl.len(), 4);
    assert_eq!(mrcl.get(0), "d");
    assert_eq!(mrcl.get(1), "c");
    assert_eq!(mrcl.get(2), "b");
    assert_eq!(mrcl.get(3), "a");
}

use lender::{ExactSizeLender, IntoLender, Lender};
use sux::dict::mapped_rear_coded_list::{MappedRearCodedListSliceU8, MappedRearCodedListStr};
use sux::traits::IntoIteratorFrom;

fn make_map(bit_width: usize, values: &[usize]) -> BitFieldVec {
    let mut bfv = BitFieldVec::new(bit_width, 0);
    for &v in values {
        bfv.push(v);
    }
    bfv
}

fn build_test_mrcl_str() -> MappedRearCodedListStr {
    let mut rclb = RearCodedListBuilder::<str, true>::new(4);
    rclb.push("aa");
    rclb.push("aab");
    rclb.push("abc");
    rclb.push("abdd");
    rclb.push("abde");
    rclb.push("abdf");
    let rcl = rclb.build();
    // Map: original[0] -> sorted[5], original[1] -> sorted[4], etc.
    let map = make_map(4, &[5, 4, 3, 2, 1, 0]); // reverse permutation
    MappedRearCodedListStr::from_parts(rcl, map)
}

fn build_test_mrcl_bytes() -> MappedRearCodedListSliceU8 {
    let mut rclb = RearCodedListBuilder::<[u8], true>::new(4);
    rclb.push(b"aa".as_slice());
    rclb.push(b"aab".as_slice());
    rclb.push(b"abc".as_slice());
    rclb.push(b"abdd".as_slice());
    let rcl = rclb.build();
    let map = make_map(3, &[3, 2, 1, 0]); // reverse permutation
    MappedRearCodedListSliceU8::from_parts(rcl, map)
}

#[test]
fn test_from_parts_str() {
    let mrcl = build_test_mrcl_str();
    assert_eq!(mrcl.len(), 6);
}

#[test]
fn test_from_parts_bytes() {
    let mrcl = build_test_mrcl_bytes();
    assert_eq!(mrcl.len(), 4);
}

#[test]
#[should_panic(expected = "Length mismatch")]
fn test_from_parts_length_mismatch() {
    let mut rclb = RearCodedListBuilder::<str, true>::new(4);
    rclb.push("aa");
    rclb.push("aab");
    let rcl = rclb.build();
    // Wrong length: 3 elements in map but only 2 in rcl
    let map = make_map(2, &[0, 1, 2]);
    let _mrcl = MappedRearCodedListStr::from_parts(rcl, map);
}

#[test]
fn test_into_parts() {
    let mrcl = build_test_mrcl_str();
    let (rcl, map) = mrcl.into_parts();
    assert_eq!(rcl.len(), 6);
    assert_eq!(value_traits::slices::SliceByValue::len(&map), 6);
}

#[test]
fn test_get_str() {
    let mrcl = build_test_mrcl_str();
    // With reverse permutation: original[0] -> sorted[5], etc.
    assert_eq!(mrcl.get(0), "abdf"); // map[0]=5 -> rcl[5]="abdf"
    assert_eq!(mrcl.get(1), "abde"); // map[1]=4 -> rcl[4]="abde"
    assert_eq!(mrcl.get(2), "abdd"); // map[2]=3 -> rcl[3]="abdd"
    assert_eq!(mrcl.get(3), "abc"); // map[3]=2 -> rcl[2]="abc"
    assert_eq!(mrcl.get(4), "aab"); // map[4]=1 -> rcl[1]="aab"
    assert_eq!(mrcl.get(5), "aa"); // map[5]=0 -> rcl[0]="aa"
}

#[test]
fn test_get_bytes() {
    let mrcl = build_test_mrcl_bytes();
    assert_eq!(mrcl.get(0), b"abdd".to_vec()); // map[0]=3 -> rcl[3]="abdd"
    assert_eq!(mrcl.get(1), b"abc".to_vec()); // map[1]=2 -> rcl[2]="abc"
    assert_eq!(mrcl.get(2), b"aab".to_vec()); // map[2]=1 -> rcl[1]="aab"
    assert_eq!(mrcl.get(3), b"aa".to_vec()); // map[3]=0 -> rcl[0]="aa"
}

#[test]
fn test_get_unchecked_str() {
    let mrcl = build_test_mrcl_str();
    unsafe {
        assert_eq!(mrcl.get_unchecked(0), "abdf");
        assert_eq!(mrcl.get_unchecked(5), "aa");
    }
}

#[test]
fn test_get_unchecked_bytes() {
    let mrcl = build_test_mrcl_bytes();
    unsafe {
        assert_eq!(mrcl.get_unchecked(0), b"abdd".to_vec());
        assert_eq!(mrcl.get_unchecked(3), b"aa".to_vec());
    }
}

#[test]
fn test_get_in_place_str() {
    let mrcl = build_test_mrcl_str();
    let mut result = String::new();
    mrcl.get_in_place(0, &mut result);
    assert_eq!(result, "abdf");
    mrcl.get_in_place(5, &mut result);
    assert_eq!(result, "aa");
}

#[test]
fn test_get_in_place_bytes() {
    let mrcl = build_test_mrcl_bytes();
    let mut result = Vec::new();
    mrcl.get_in_place(0, &mut result);
    assert_eq!(result, b"abdd");
    mrcl.get_in_place(3, &mut result);
    assert_eq!(result, b"aa");
}

#[test]
fn test_get_bytes_str() {
    let mrcl = build_test_mrcl_str();
    assert_eq!(mrcl.get_bytes(0), b"abdf".to_vec());
    assert_eq!(mrcl.get_bytes(5), b"aa".to_vec());
}

#[test]
fn test_get_bytes_in_place_str() {
    let mrcl = build_test_mrcl_str();
    let mut result = Vec::new();
    mrcl.get_bytes_in_place(0, &mut result);
    assert_eq!(result, b"abdf");
    mrcl.get_bytes_in_place(5, &mut result);
    assert_eq!(result, b"aa");
}

#[test]
fn test_iter_str() {
    let mrcl = build_test_mrcl_str();
    let items: Vec<String> = mrcl.iter().collect();
    assert_eq!(items.len(), 6);
    assert_eq!(items[0], "abdf");
    assert_eq!(items[5], "aa");
}

#[test]
fn test_iter_bytes() {
    let mrcl = build_test_mrcl_bytes();
    let items: Vec<Vec<u8>> = mrcl.iter().collect();
    assert_eq!(items.len(), 4);
    assert_eq!(items[0], b"abdd".to_vec());
    assert_eq!(items[3], b"aa".to_vec());
}

#[test]
fn test_iter_from_str() {
    let mrcl = build_test_mrcl_str();
    let items: Vec<String> = mrcl.iter_from(3).collect();
    assert_eq!(items.len(), 3);
    assert_eq!(items[0], "abc"); // position 3 -> map[3]=2 -> rcl[2]="abc"
    assert_eq!(items[2], "aa"); // position 5 -> map[5]=0 -> rcl[0]="aa"
}

#[test]
fn test_iter_from_bytes() {
    let mrcl = build_test_mrcl_bytes();
    let items: Vec<Vec<u8>> = mrcl.iter_from(2).collect();
    assert_eq!(items.len(), 2);
    assert_eq!(items[0], b"aab".to_vec());
    assert_eq!(items[1], b"aa".to_vec());
}

#[test]
fn test_iter_exact_size() {
    let mrcl = build_test_mrcl_str();
    let iter = mrcl.iter();
    assert_eq!(iter.len(), 6);

    let iter_from = mrcl.iter_from(3);
    assert_eq!(iter_from.len(), 3);
}

#[test]
fn test_iter_size_hint() {
    let mrcl = build_test_mrcl_str();
    let iter = mrcl.iter();
    assert_eq!(iter.size_hint(), (6, Some(6)));
}

#[test]
fn test_lender_str() {
    let mrcl = build_test_mrcl_str();
    let mut lender = mrcl.lender();

    assert_eq!(lender.next(), Some("abdf"));
    assert_eq!(lender.next(), Some("abde"));
    assert_eq!(lender.next(), Some("abdd"));
    assert_eq!(lender.next(), Some("abc"));
    assert_eq!(lender.next(), Some("aab"));
    assert_eq!(lender.next(), Some("aa"));
    assert_eq!(lender.next(), None);
}

#[test]
fn test_lender_bytes() {
    let mrcl = build_test_mrcl_bytes();
    let mut lender = mrcl.lender();

    assert_eq!(lender.next(), Some(b"abdd".as_slice()));
    assert_eq!(lender.next(), Some(b"abc".as_slice()));
    assert_eq!(lender.next(), Some(b"aab".as_slice()));
    assert_eq!(lender.next(), Some(b"aa".as_slice()));
    assert_eq!(lender.next(), None);
}

#[test]
fn test_lender_from_str() {
    let mrcl = build_test_mrcl_str();
    let mut lender = mrcl.lender_from(4);

    assert_eq!(lender.next(), Some("aab"));
    assert_eq!(lender.next(), Some("aa"));
    assert_eq!(lender.next(), None);
}

#[test]
fn test_lender_from_bytes() {
    let mrcl = build_test_mrcl_bytes();
    let mut lender = mrcl.lender_from(2);

    assert_eq!(lender.next(), Some(b"aab".as_slice()));
    assert_eq!(lender.next(), Some(b"aa".as_slice()));
    assert_eq!(lender.next(), None);
}

#[test]
fn test_lender_exact_size() {
    let mrcl = build_test_mrcl_str();
    let lender = mrcl.lender();
    assert_eq!(ExactSizeLender::len(&lender), 6);

    let lender_from = mrcl.lender_from(4);
    assert_eq!(ExactSizeLender::len(&lender_from), 2);
}

#[test]
fn test_lender_size_hint() {
    let mrcl = build_test_mrcl_str();
    let lender = mrcl.lender();
    assert_eq!(lender.size_hint(), (6, Some(6)));
}

#[test]
fn test_into_lender() {
    let mrcl = build_test_mrcl_str();
    let mut lender = (&mrcl).into_lender();

    assert_eq!(lender.next(), Some("abdf"));
    assert_eq!(lender.next(), Some("abde"));
}

#[test]
fn test_into_iterator() {
    let mrcl = build_test_mrcl_str();
    let items: Vec<String> = (&mrcl).into_iter().collect();
    assert_eq!(items.len(), 6);
    assert_eq!(items[0], "abdf");
}

#[test]
fn test_into_iterator_from() {
    let mrcl = build_test_mrcl_str();
    let items: Vec<String> = (&mrcl).into_iter_from(4).collect();
    assert_eq!(items.len(), 2);
    assert_eq!(items[0], "aab");
    assert_eq!(items[1], "aa");
}

#[test]
fn test_identity_map() {
    // Test with an identity mapping (no reordering)
    let mut rclb = RearCodedListBuilder::<str, true>::new(4);
    rclb.push("aa");
    rclb.push("aab");
    rclb.push("abc");
    let rcl = rclb.build();
    let map = make_map(2, &[0, 1, 2]); // identity
    let mrcl = MappedRearCodedListStr::from_parts(rcl, map);

    assert_eq!(mrcl.get(0), "aa");
    assert_eq!(mrcl.get(1), "aab");
    assert_eq!(mrcl.get(2), "abc");
}

#[test]
fn test_lender_fused() {
    let mrcl = build_test_mrcl_str();
    let mut lender = mrcl.lender();

    // Exhaust the lender
    while lender.next().is_some() {}

    // FusedLender should continue returning None
    assert!(lender.next().is_none());
    assert!(lender.next().is_none());
    assert!(lender.next().is_none());
}

#[test]
fn test_empty() {
    let rclb = RearCodedListBuilder::<str, true>::new(4);
    let rcl = rclb.build();
    let map: BitFieldVec = BitFieldVec::new(1, 0); // empty map
    let mrcl = MappedRearCodedListStr::from_parts(rcl, map);

    assert_eq!(mrcl.len(), 0);
    assert!(mrcl.iter().next().is_none());
    assert!(mrcl.lender().next().is_none());
}

#[test]
fn test_single_element() {
    let mut rclb = RearCodedListBuilder::<str, true>::new(4);
    rclb.push("hello");
    let rcl = rclb.build();
    let map = make_map(1, &[0]); // identity for single element
    let mrcl = MappedRearCodedListStr::from_parts(rcl, map);

    assert_eq!(mrcl.len(), 1);
    assert_eq!(mrcl.get(0), "hello");
    assert_eq!(mrcl.iter().collect::<Vec<_>>(), vec!["hello".to_string()]);
}
