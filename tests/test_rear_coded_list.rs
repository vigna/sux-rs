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
    use epserde::Epserde;
    use epserde::deser::Deserialize;
    use epserde::prelude::SerIter;
    use epserde::ser::Serialize;
    use epserde::utils::AlignedCursor;
    use indexed_dict::*;
    use lender::*;
    use rand::prelude::*;
    use std::cell::Cell;
    use std::cell::RefCell;
    use std::io::BufReader;
    use std::io::prelude::*;
    use std::sync::Mutex;
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
            <RearCodedList>::read_mmap(&mut cursor, len, epserde::deser::Flags::empty())?
        };
        let c = c.uncase();

        for (i, word) in words.iter().enumerate() {
            assert_eq!(&c.get(i), word);
        }

        // test unsorted RCL

        let mut rcl_builder = <RearCodedListBuilder<false>>::new(4);
        let mut shuffled_words = words.iter().map(|s| s.as_str()).collect::<Vec<_>>();
        shuffled_words.shuffle(&mut rand::rng());

        for string in shuffled_words.iter() {
            rcl_builder.push(string);
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
            <RearCodedList<Box<[u8]>, Box<[usize]>, false>>::read_mmap(
                &mut cursor,
                len,
                epserde::deser::Flags::empty(),
            )?
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
        let mut rcab = <RearCodedListBuilder>::new(4);
        rcab.push("apple");
        rcab.push("banana");
        rcab.push("cherry");
        // This should panic because "apricot" < "cherry"
        rcab.push("apricot");
    }

    #[test]
    fn test_concept() {
        use epserde::utils::AlignedCursor;
        use maligned::A16;
        #[derive(Epserde, Debug)]
        struct Struct<S, L> {
            strings: S,
            lengths: L,
        }

        let strings = ["a", "cb", "ccc"];
        let iter = strings.iter();
        let mut cursor = <AlignedCursor<A16>>::new();

        struct IterState<'a, I: Iterator<Item = &'a &'static str> + ExactSizeIterator> {
            iter: I,
            lengths: Vec<usize>,
            pos: usize,
        }

        let state = RefCell::new(IterState {
            iter,
            lengths: Vec::new(),
            pos: 0,
        });

        struct WritingIter<'a, 'b, I: Iterator<Item = &'a &'static str> + ExactSizeIterator> {
            state: &'b RefCell<IterState<'a, I>>,
        }

        impl<'a, 'b, I: Iterator<Item = &'a &'static str> + ExactSizeIterator> Iterator
            for WritingIter<'a, 'b, I>
        {
            type Item = &'a &'static str;

            fn next(&mut self) -> Option<Self::Item> {
                let mut state = self.state.borrow_mut();
                if let Some(s) = state.iter.next() {
                    state.lengths.push(s.len());
                    Some(s)
                } else {
                    None
                }
            }
        }

        impl<'a, 'b, I: Iterator<Item = &'a &'static str> + ExactSizeIterator> ExactSizeIterator
            for WritingIter<'a, 'b, I>
        {
            fn len(&self) -> usize {
                self.state.borrow().iter.len()
            }
        }

        let writing_iter = WritingIter { state: &state };

        let s = Struct {
            strings: SerIter::<&'static str, _>::new(writing_iter),
            lengths: SerIter::new(IterFromDelayedMutex::new(&state)),
        };

        unsafe { s.serialize(&mut cursor).unwrap() };
        cursor.set_position(0);
        let t = unsafe { Struct::<Box<[String]>, Box<[usize]>>::deserialize_full(&mut cursor) }
            .unwrap();
        dbg!(t);

        pub struct IterFromDelayedMutex<
            'a,
            'b,
            I: Iterator<Item = &'a &'static str> + ExactSizeIterator,
        > {
            state: &'b RefCell<IterState<'a, I>>,
        }

        impl<'a, 'b, I: Iterator<Item = &'a &'static str> + ExactSizeIterator>
            IterFromDelayedMutex<'a, 'b, I>
        {
            pub fn new(state: &'b RefCell<IterState<'a, I>>) -> Self {
                Self { state }
            }
        }

        impl<'a, 'b, I: Iterator<Item = &'a &'static str> + ExactSizeIterator> Iterator
            for IterFromDelayedMutex<'a, 'b, I>
        {
            type Item = usize;

            fn next(&mut self) -> Option<Self::Item> {
                let mut borrow = self.state.borrow_mut();
                if borrow.lengths.len() == borrow.pos {
                    None
                } else {
                    let item = borrow.lengths[borrow.pos];
                    borrow.pos += 1;
                    Some(item)
                }
            }
        }

        impl<'a, 'b, I: Iterator<Item = &'a &'static str> + ExactSizeIterator> ExactSizeIterator
            for IterFromDelayedMutex<'a, 'b, I>
        {
            fn len(&self) -> usize {
                let borrow = self.state.borrow();
                borrow.lengths.len() - borrow.pos
            }
        }
    }
}
