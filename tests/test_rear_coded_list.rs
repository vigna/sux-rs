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

        struct WritingIter<'a, I: Iterator<Item = &'a &'static str> + ExactSizeIterator> {
            strings: I,
            lengths: Vec<usize>,
            resulting_lengths: &'a Mutex<Option<Vec<usize>>>,
        }

        impl<'a, I: Iterator<Item = &'a &'static str> + ExactSizeIterator> Iterator for WritingIter<'a, I> {
            type Item = &'a &'static str;

            fn next(&mut self) -> Option<Self::Item> {
                if let Some(s) = self.strings.next() {
                    self.lengths.push(s.len());
                    Some(s)
                } else {
                    let mut lengths = Vec::new();
                    std::mem::swap(&mut lengths, &mut self.lengths);
                    *self.resulting_lengths.lock().unwrap() = Some(lengths);
                    None
                }
            }
        }

        impl<'a, I: Iterator<Item = &'a &'static str> + ExactSizeIterator> ExactSizeIterator
            for WritingIter<'a, I>
        {
            fn len(&self) -> usize {
                self.strings.len()
            }
        }

        let lengths = Mutex::new(None);
        let writing_iter = WritingIter {
            strings: iter,
            lengths: Vec::new(),
            resulting_lengths: &lengths,
        };
        let s = Struct {
            strings: SerIter::<&'static str, _>::new(writing_iter),
            lengths: SerIter::new(IterFromDelayedMutex::new(&lengths)),
        };

        unsafe { s.serialize(&mut cursor).unwrap() };
        cursor.set_position(0);
        let t = unsafe { Struct::<Box<[String]>, Box<[usize]>>::deserialize_full(&mut cursor) }
            .unwrap();
        dbg!(t);
    }

    pub struct IterFromDelayedMutex<'a, T: IntoIterator> {
        /// store the IntoIterator before we start using it
        from_lock: &'a Mutex<Option<T>>,
        /// store the iterator between a call to .len() (which can't mutate self.iter) and a call
        /// to .next() (which moves it from self.iter_mutex to self.iter to avoid unnecessary
        /// synchronization)
        iter_mutex: Mutex<Option<T::IntoIter>>,
        /// store the iterator after we called .next() on it
        iter: Option<T::IntoIter>,
    }
    impl<'a, T: IntoIterator> IterFromDelayedMutex<'a, T> {
        fn new(from_lock: &'a Mutex<Option<T>>) -> Self {
            Self {
                from_lock: from_lock,
                iter_mutex: Mutex::new(None),
                iter: None,
            }
        }
    }

    impl<'a, T: IntoIterator> Iterator for IterFromDelayedMutex<'a, T> {
        type Item = <T::IntoIter as Iterator>::Item;

        fn next(&mut self) -> Option<Self::Item> {
            // if self.iter is not set yet, get it from self.from_lock or self.iter_mutex,
            // and set self.iter
            self.iter
                .get_or_insert_with(|| {
                    self.iter_mutex.lock().unwrap().take().unwrap_or_else(|| {
                        self.from_lock
                            .lock()
                            .unwrap()
                            .take()
                            .expect("from_lock is empty")
                            .into_iter()
                    })
                })
                .next()
        }
    }

    impl<'a, T: IntoIterator<IntoIter: ExactSizeIterator>> ExactSizeIterator
        for IterFromDelayedMutex<'a, T>
    {
        fn len(&self) -> usize {
            // if self.iter is not set yet, look at self.iter_mutex. if neither is set, then set
            // the latter from self.from_lock
            self.iter
                .as_ref()
                .map(|iter| iter.len())
                .unwrap_or_else(|| {
                    self.iter_mutex
                        .lock()
                        .unwrap()
                        .get_or_insert_with(|| {
                            self.from_lock
                                .lock()
                                .unwrap()
                                .take()
                                .expect("from_lock is empty")
                                .into_iter()
                        })
                        .len()
                })
        }
    }
}
