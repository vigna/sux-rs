/*
 *
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

/*!

Fast sorting of signatures and values.
*/

use anyhow::{bail, Result};
use rayon::slice::ParallelSliceMut;
use std::{fs::File, io::*};

/// Adapter to iterate over the lines of a file.
pub struct SigSorter {
    num_keys: usize,
    buf_high_bits: u32,
    writers: Vec<BufWriter<File>>,
    buf_sizes: Vec<usize>,
}

pub struct SortedSig {
    buf_high_bits: u32,
    chunk_high_bits: u32,
    files: Vec<File>,
    buf_sizes: Vec<usize>,
    chunk_sizes: Vec<usize>,
    chunk: usize,
}

impl SortedSig {
    pub fn counts(&self) -> &[usize] {
        &self.chunk_sizes
    }
}

impl Iterator for SortedSig {
    type Item = (usize, Box<[([u64; 2], u64)]>);
    fn next(&mut self) -> Option<Self::Item> {
        if self.files.is_empty() {
            None
        } else {
            let mut data = vec![([0_u64; 2], 0_u64); self.chunk_sizes.remove(0)];

            if self.buf_high_bits >= self.chunk_high_bits {
                let to_aggr = 1 << (self.buf_high_bits - self.chunk_high_bits);

                {
                    let (pre, mut buf, after) = unsafe { data.align_to_mut::<u8>() };
                    assert!(pre.is_empty());
                    assert!(after.is_empty());
                    for _ in 0..to_aggr {
                        let mut reader = self.files.remove(0);
                        let bytes =
                            self.buf_sizes.remove(0) * core::mem::size_of::<([u64; 2], u64)>();
                        reader.read_exact(&mut buf[..bytes]).unwrap();
                        buf = &mut buf[bytes..];
                    }
                }

                let res = Some((self.chunk, data.into_boxed_slice()));
                self.chunk += 1;
                res
            } else {
                {
                    let (pre, buf, after) = unsafe { data.align_to_mut::<u8>() };
                    assert!(pre.is_empty());
                    assert!(after.is_empty());
                    self.files[0].read_exact(buf).unwrap();
                }

                let res = Some((self.chunk, data.into_boxed_slice()));
                self.chunk += 1;
                if self.chunk % (1 << (self.chunk_high_bits - self.buf_high_bits)) == 0 {
                    self.files.remove(0);
                }
                res
            }
        }
    }
}

impl SigSorter {
    pub fn new(buf_high_bits: u32) -> Result<Self> {
        let temp_dir = tempfile::TempDir::new()?;
        let mut writers = vec![];
        for i in 0..1 << buf_high_bits {
            let file = File::options()
                .read(true)
                .write(true)
                .create(true)
                .open(temp_dir.path().join(format!("{}.tmp", i)))?;
            writers.push(BufWriter::new(file));
        }
        Ok(Self {
            num_keys: 0,
            buf_high_bits,
            writers,
            buf_sizes: vec![0; 1 << buf_high_bits],
        })
    }

    pub fn push(&mut self, value: &([u64; 2], u64)) -> Result<()> {
        self.num_keys += 1;
        // high_bits can be 0
        let buffer =
            (value.0[0].rotate_left(self.buf_high_bits)) as usize & ((1 << self.buf_high_bits) - 1);
        self.buf_sizes[buffer] += 1;
        self.writers[buffer].write_all(&value.0[0].to_ne_bytes())?;
        self.writers[buffer].write_all(&value.0[1].to_ne_bytes())?;
        Ok(self.writers[buffer].write_all(&value.1.to_ne_bytes())?)
    }

    pub fn extend(&mut self, iter: impl IntoIterator<Item = ([u64; 2], u64)>) -> Result<()> {
        for value in iter {
            self.push(&value)?;
        }
        Ok(())
    }

    pub fn num_keys(&self) -> usize {
        self.num_keys
    }

    pub fn sort(mut self, chunk_high_bits: u32) -> Result<SortedSig> {
        let mut max_len = 0;
        let mut lens = vec![];
        let mut files = vec![];

        for _ in 0..1 << self.buf_high_bits {
            self.writers[0].flush()?; // We will remove it

            let len = self.writers[0].stream_position()?;
            lens.push(len);
            max_len = max_len.max(len);

            let mut file = self.writers.remove(0).into_inner()?;
            file.seek(SeekFrom::Start(0))?;
            files.push(file);
        }

        let mut buf = vec![0_u8; max_len as usize];
        let mut counts = vec![0; 1 << chunk_high_bits];
        for i in 0..1 << self.buf_high_bits {
            files[i].read_exact(&mut buf[..lens[i] as usize])?;
            {
                let (pre, data, after) =
                    unsafe { buf[..lens[i] as usize].align_to_mut::<([u64; 2], u64)>() };
                assert!(pre.is_empty());
                assert!(after.is_empty());
                data.par_sort_unstable();

                counts[data[0].0[0].rotate_left(chunk_high_bits) as usize
                    & ((1 << chunk_high_bits) - 1)] += 1;

                for w in data.windows(2) {
                    counts[w[1].0[0].rotate_left(chunk_high_bits) as usize
                        & ((1 << chunk_high_bits) - 1)] += 1;
                    if w[0].0 == w[1].0 {
                        bail!("Duplicate key");
                    }
                }
            }
            files[i].seek(SeekFrom::Start(0))?;
            files[i].write_all(&buf[..lens[i] as usize])?;
            files[i].flush()?;
            files[i].seek(SeekFrom::Start(0))?;
        }

        Ok(SortedSig {
            buf_high_bits: self.buf_high_bits,
            chunk_high_bits,
            files,
            buf_sizes: self.buf_sizes,
            chunk_sizes: counts,
            chunk: 0,
        })
    }
}

#[test]

fn test_sig_sorter() {
    use rand::prelude::*;
    for high_bits in [0, 2, 4] {
        for chunk_bits in [0, 2, 4] {
            let mut sig_sorter = SigSorter::new(high_bits).unwrap();
            let mut rand = SmallRng::seed_from_u64(0);
            for _ in (0..1000).rev() {
                sig_sorter
                    .push(&([rand.next_u64(), rand.next_u64()], rand.next_u64()))
                    .unwrap();
            }
            let sorted_sig = sig_sorter.sort(chunk_bits).unwrap();
            let mut count = 0;
            for chunk in sorted_sig {
                count += 1;
                for w in chunk.1.windows(2) {
                    assert!(
                        w[0].0[0] < w[1].0[0] || w[0].0[0] == w[1].0[0] && w[0].0[1] < w[1].0[1]
                    );
                }
            }
            assert_eq!(count, 1 << chunk_bits);
        }
    }
}
