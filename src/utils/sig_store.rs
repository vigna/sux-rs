/*
 *
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

/*!

Fast sorting and grouping of signatures and values.

A *signature* is a pair of 64-bit integers, and a *value* is a 64-bit integer.
A [`SigStore`] accepts signatures and values in any order; then,
when you call [`SigStore::into_iter`] you can specify the number of high bits
to use for grouping signatures into chunks.

*/

use anyhow::{bail, Result};
use rayon::slice::ParallelSliceMut;
use std::{
    fmt::{Display, Formatter},
    fs::File,
    io::*,
};

/**

This structure is used to sort key signatures (i.e., randomly-looking
hashes associated to keys) and associated values in a fast way,
and to group them by the high bits of the hash. It accepts signatures and values
in any order, and it sorts them in a way that allows to group them into *chunks*
using their highest bits.

The implementation exploits the fact that signatures are randomly distributed,
and thus bucket sorting is very effective: at construction time you specify
the number of high bits to use for bucket sorting (say, 8), and when you
[push](`SigStore::push`) keys they will be stored in different disk buffers
(in this case, 256) depending on their high bits. The buffer will be stored
in a directory created by [`tempfile::TempDir`].

When you call [`SigStore::into_iter`] you can specify the number of high bits
to use for grouping signatures into chunks, and the necessary buffer splitting or merging
will be handled automatically.

*/
pub struct SigStore {
    num_keys: usize,
    buckets_high_bits: u32,
    writers: Vec<BufWriter<File>>,
    buf_sizes: Vec<usize>,
}

/**

The iterator on chunks returned by [`SigStore::into_iter`].

*/
#[derive(Debug)]
pub struct Chunks {
    buf_high_bits: u32,
    chunk_high_bits: u32,
    files: Vec<File>,
    buf_sizes: Vec<usize>,
    chunk_sizes: Vec<usize>,
    chunk: usize,
}

impl Chunks {
    /// The sizes of the chunks returned by the iterator.
    pub fn counts(&self) -> &[usize] {
        &self.chunk_sizes
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct DuplicateSigError {}

impl Display for DuplicateSigError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Duplicate signature detected")
    }
}

impl std::error::Error for DuplicateSigError {}

impl Iterator for Chunks {
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

impl SigStore {
    /// Create a new store with 2<sup>`buf_high_bits`</sup> buffers.
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
            buckets_high_bits: buf_high_bits,
            writers,
            buf_sizes: vec![0; 1 << buf_high_bits],
        })
    }

    /// Adds a pair of signatures and values to the store.
    pub fn push(&mut self, value: &([u64; 2], u64)) -> Result<()> {
        self.num_keys += 1;
        // high_bits can be 0
        let buffer = (value.0[0].rotate_left(self.buckets_high_bits)) as usize
            & ((1 << self.buckets_high_bits) - 1);
        self.buf_sizes[buffer] += 1;
        self.writers[buffer].write_all(&value.0[0].to_ne_bytes())?;
        self.writers[buffer].write_all(&value.0[1].to_ne_bytes())?;
        Ok(self.writers[buffer].write_all(&value.1.to_ne_bytes())?)
    }

    /// Adds pairs of signature and value to the store.
    pub fn extend(&mut self, iter: impl IntoIterator<Item = ([u64; 2], u64)>) -> Result<()> {
        for value in iter {
            self.push(&value)?;
        }
        Ok(())
    }

    /// The number of keys added to the store so far.
    pub fn num_keys(&self) -> usize {
        self.num_keys
    }

    /// Sorts the signatures and values and returns
    /// an iterator on 2<sup>`chunk_high_bits`</sup> chunks grouped
    /// by the highest `chunk_high_bits` bits of the signatures.
    ///
    /// Beside I/O error, this method might return a [`DuplicateKeyError`].
    pub fn into_iter(mut self, chunk_high_bits: u32) -> Result<Chunks> {
        let mut max_len = 0;
        let mut lens = vec![];
        let mut files = vec![];

        for _ in 0..1 << self.buckets_high_bits {
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
        for i in 0..1 << self.buckets_high_bits {
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
                        bail!(DuplicateSigError {});
                    }
                }
            }
            files[i].seek(SeekFrom::Start(0))?;
            files[i].write_all(&buf[..lens[i] as usize])?;
            files[i].flush()?;
            files[i].seek(SeekFrom::Start(0))?;
        }

        Ok(Chunks {
            buf_high_bits: self.buckets_high_bits,
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
            let mut sig_sorter = SigStore::new(high_bits).unwrap();
            let mut rand = SmallRng::seed_from_u64(0);
            for _ in (0..1000).rev() {
                sig_sorter
                    .push(&([rand.next_u64(), rand.next_u64()], rand.next_u64()))
                    .unwrap();
            }
            let sorted_sig = sig_sorter.into_iter(chunk_bits).unwrap();
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

#[test]

fn test_dup() {
    let mut sig_sorter = SigStore::new(0).unwrap();
    sig_sorter.push(&([0, 0], 0)).unwrap();
    sig_sorter.push(&([0, 0], 0)).unwrap();
    assert!(sig_sorter
        .into_iter(0)
        .unwrap_err()
        .downcast_ref::<DuplicateSigError>()
        .is_some());
}
