/*
 *
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

/*!

Fast sorting of signatures and values.
*/

use anyhow::Result;
use rayon::slice::ParallelSliceMut;
use std::{fs::File, io::*};
use tempfile::TempDir;

/// Adapter to iterate over the lines of a file.
pub struct SigSorter {
    high_bits: u32,
    writers: Vec<BufWriter<File>>,
    tmp_dir: TempDir,
}

pub struct SortedSig {
    pub num_keys: usize,
    readers: Vec<BufReader<File>>,
    counts: Vec<usize>,
    chunk: usize,
}

impl SortedSig {
    pub fn counts(&self) -> &[usize] {
        &self.counts
    }
}

impl Iterator for SortedSig {
    type Item = (usize, Box<[([u64; 2], u64)]>);
    fn next(&mut self) -> Option<Self::Item> {
        if self.readers.is_empty() {
            None
        } else {
            let mut reader = self.readers.remove(0);
            let count = self.counts.remove(0);
            let mut data = vec![([0_u64; 2], 0_u64); count];
            {
                let (pre, mut buf, after) = unsafe { data.align_to_mut::<u8>() };
                assert!(pre.is_empty());
                assert!(after.is_empty());
                reader.read_exact(&mut buf).unwrap();
            }

            let res = Some((self.chunk, data.into_boxed_slice()));
            self.chunk += 1;
            res
        }
    }
}

impl SigSorter {
    pub fn new(high_bits: u32) -> Result<Self> {
        let tmp_dir = tempfile::TempDir::new().unwrap();
        let mut writers = vec![];
        for i in 0..1 << high_bits {
            let file = File::options()
                .read(true)
                .write(true)
                .create(true)
                .open(tmp_dir.path().join(format!("{}.tmp", i)))?;
            writers.push(BufWriter::new(file));
        }
        Ok(Self {
            high_bits,
            writers,
            tmp_dir,
        })
    }

    pub fn push(&mut self, value: &([u64; 2], u64)) -> Result<()> {
        // high_bits can be 0
        let chunk = (value.0[0].rotate_left(self.high_bits)) as usize & ((1 << self.high_bits) - 1);
        self.writers[chunk].write_all(&value.0[0].to_ne_bytes())?;
        self.writers[chunk].write_all(&value.0[1].to_ne_bytes())?;
        Ok(self.writers[chunk].write_all(&value.1.to_ne_bytes())?)
    }

    pub fn extend(&mut self, iter: impl IntoIterator<Item = ([u64; 2], u64)>) -> Result<()> {
        for value in iter {
            self.push(&value)?;
        }
        Ok(())
    }

    pub fn sort(mut self) -> Result<SortedSig> {
        let mut max_len = 0;
        let mut lens = vec![];
        let mut files = vec![];
        let mut max_count = 0;
        for _ in (0..1 << self.high_bits).rev() {
            self.writers[0].flush()?; // We will remove it

            let len = self.writers[0].stream_position()?;
            lens.push(len);
            max_len = max_len.max(len);

            let mut file = self.writers.remove(0).into_inner()?;
            file.seek(SeekFrom::Start(0))?;
            files.push(file);
        }

        let mut buf = vec![0_u8; max_len as usize];
        let mut counts = vec![0; 1 << self.high_bits];
        for i in 0..1 << self.high_bits {
            files[i].read_exact(&mut buf[..lens[i] as usize])?;
            {
                let (pre, data, after) =
                    unsafe { buf[..lens[i] as usize].align_to_mut::<([u64; 2], u64)>() };
                assert!(pre.is_empty());
                assert!(after.is_empty());
                data.par_sort_unstable();
                max_count = max_count.max(data.len());
                counts[i] = data.len();

                for w in data.windows(2) {
                    if w[0].0 == w[1].0 {
                        return Err(anyhow::anyhow!("duplicate signature"));
                    }
                }
            }
            files[i].seek(SeekFrom::Start(0))?;
            files[i].write_all(&buf[..lens[i] as usize])?;
            files[i].flush()?;
        }

        let mut readers = vec![];
        for file in files {
            let mut reader = BufReader::new(file);
            reader.seek(SeekFrom::Start(0))?;
            readers.push(reader);
        }

        dbg!(readers.len(), &counts);
        Ok(SortedSig {
            num_keys: counts.iter().sum(),
            readers,
            counts,
            chunk: 0,
        })
    }
}

#[test]

fn test_sig_sorter() {
    use rand::prelude::*;
    let mut sig_sorter = SigSorter::new(4).unwrap();
    let mut rand = SmallRng::seed_from_u64(0);
    for _ in (0..1000).rev() {
        sig_sorter
            .push(&([rand.next_u64(), rand.next_u64()], rand.next_u64()))
            .unwrap();
    }
    let sorted_sig = sig_sorter.sort().unwrap();
    for chunk in sorted_sig {
        for w in chunk.1.windows(2) {
            assert!(w[0].0[0] < w[1].0[0] || w[0].0[0] == w[1].0[0] && w[0].0[1] < w[1].0[1]);
        }
    }
}
