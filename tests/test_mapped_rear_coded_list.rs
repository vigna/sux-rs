/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 * SPDX-FileCopyrightText: 2023 Tommaso Fontana
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

#[cfg(feature = "epserde")]
mod test {
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
        use maligned::A16;
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

        let mut cursor = <AlignedCursor<A16>>::new();
        let schema = unsafe { mrcl.serialize_with_schema(&mut cursor)? };
        println!("{}", schema.to_csv());

        let cursor_len = cursor.len();
        cursor.set_position(0);
        let c = unsafe {
            <MappedRearCodedListStr>::read_mmap(
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

        let mut cursor = <AlignedCursor<A16>>::new();
        let schema = unsafe { mrcl.serialize_with_schema(&mut cursor)? };
        println!("{}", schema.to_csv());

        let cursor_len = cursor.len();
        cursor.set_position(0);
        let c = unsafe {
            <MappedRearCodedListStr<false>>::read_mmap(
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
