/*
 *
 * SPDX-FileCopyrightText: 2023 Tommaso Fontana
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Ported from <https://github.com/vigna/Sux4J/blob/master/c/mph.c>

use super::spooky::{spooky_short, spooky_short_rehash};
use anyhow::Result;
use std::fs::File;
use std::io::BufReader;
use std::io::Read;
use std::path::Path;

/// A stub to load and access Genuzio-Ottaviano-Vigna minimal perfect hash functions.
///
/// To generate the structure you must use the Java version:
/// ```bash
/// java it.unimi.dsi.sux4j.mph.GOVMinimalPerfectHashFunction --byte-array SOURCE test.mph
/// ```
///
/// To obtain a file that can be read by this structure, load
/// the serialized Java instance of the minimal perfect hash function and
/// dump it in C-compatible format:
/// ```shell
/// echo '((it.unimi.dsi.sux4j.mph.GOVMinimalPerfectHashFunction)it.unimi.dsi.fastutil.io.BinIO.loadObject("test.mph")).dump("test.cmph");' | jshell
/// ```
///
/// You can now load the dumped file with the [`load`](crate::mph::gov::GOVMPH::load) method.
///
/// # Reference:
/// - [Marco Genuzio, Giuseppe Ottaviano, and Sebastiano Vigna, Fast Scalable Construction of (Minimal Perfect Hash) Functions](https://arxiv.org/pdf/1603.04330.pdf)
/// - [Java version with `dump` method](https://github.com/vigna/Sux4J/blob/master/src/it/unimi/dsi/sux4j/mph/GOVMinimalPerfectHashFunction.java)
#[derive(Debug)]
pub struct GOVMPH {
    pub size: u64,
    pub multiplier: u64,
    pub global_seed: u64,
    pub edge_offset_and_seed: Vec<u64>,
    pub array: Vec<u64>,
}

impl GOVMPH {
    pub fn size(&self) -> u64 {
        self.size
    }

    pub fn get_byte_array(&self, key: &[u8]) -> u64 {
        let signature = spooky_short(key, self.global_seed);
        let bucket = ((((signature[0] as u128) >> 1) * (self.multiplier as u128)) >> 64) as u64;
        let edge_offset_seed = self.edge_offset_and_seed[bucket as usize];
        let bucket_offset = vertex_offset(edge_offset_seed);
        let num_variables =
            vertex_offset(self.edge_offset_and_seed[bucket as usize + 1]) - bucket_offset;
        let e = signature_to_equation(&signature, edge_offset_seed & (!OFFSET_MASK), num_variables);
        let eq_idx = (get_2bit_value(&self.array, e[0] + bucket_offset)
            + get_2bit_value(&self.array, e[1] + bucket_offset)
            + get_2bit_value(&self.array, e[2] + bucket_offset))
            % 3;
        let offset = count_nonzero_pairs(
            bucket_offset,
            bucket_offset + e[eq_idx as usize],
            &self.array,
        );
        (edge_offset_seed & OFFSET_MASK) + offset
    }
}

#[inline(always)]
#[must_use]
/// Count the number of pairs of bits that are both set in a word.
const fn count_non_zero_pairs_in_word(x: u64) -> u64 {
    ((x | x >> 1) & 0x5555555555555555).count_ones() as u64
}

/// Count the number of pairs of bits that are both set in a slice of words from
/// bit offset `start` to bit offset `end`.
fn count_nonzero_pairs(start: u64, end: u64, array: &[u64]) -> u64 {
    let mut block = start / 32;
    let end_block = end / 32;
    let start_offset = start % 32;
    let end_offset = end % 32;

    if block == end_block {
        return count_non_zero_pairs_in_word(
            (array[block as usize] & ((1 << (end_offset * 2)) - 1)) >> (start_offset * 2),
        );
    }

    let mut pairs = 0;
    if start_offset != 0 {
        pairs += count_non_zero_pairs_in_word(array[block as usize] >> (start_offset * 2));
        block += 1;
    }
    while block < end_block {
        pairs += count_non_zero_pairs_in_word(array[block as usize]);
        block += 1;
    }
    if end_offset != 0 {
        pairs +=
            count_non_zero_pairs_in_word(array[block as usize] & ((1 << (end_offset * 2)) - 1));
    }
    pairs
}

const OFFSET_MASK: u64 = u64::MAX >> 8;
const C_TIMES_256: u64 = 281; // floor((1.09 + 0.01) * 256.0)

#[inline(always)]
#[must_use]
fn signature_to_equation(signature: &[u64; 4], seed: u64, num_variables: u64) -> [u64; 3] {
    let hash = spooky_short_rehash(signature, seed);
    let shift = num_variables.leading_zeros();
    let mask = (1_u64 << shift) - 1;
    [
        ((hash[0] & mask) * num_variables) >> shift,
        ((hash[1] & mask) * num_variables) >> shift,
        ((hash[2] & mask) * num_variables) >> shift,
    ]
}

#[inline(always)]
#[must_use]
const fn vertex_offset(edge_offset_seed: u64) -> u64 {
    ((edge_offset_seed & OFFSET_MASK) * C_TIMES_256) >> 8
}

#[inline(always)]
#[must_use]
const fn get_2bit_value(data: &[u64], mut pos: u64) -> u64 {
    pos *= 2;
    (data[(pos / 64) as usize] >> (pos % 64)) & 3
}

impl GOVMPH {
    /// Given a path to a file `.cmph` generated from the java version,
    /// load a GOVMPH structure from a file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::load_reader(BufReader::new(File::open(path.as_ref())?))
    }

    /// Given a generic `Read` implementor, load a GOVMPH structure from a file.
    pub fn load_reader<F: Read>(mut file: F) -> Result<Self> {
        macro_rules! read {
            ($file:expr, $type:ty) => {{
                let mut buffer: [u8; core::mem::size_of::<$type>()] =
                    [0; core::mem::size_of::<$type>()];
                $file.read_exact(&mut buffer)?;
                <$type>::from_le_bytes(buffer)
            }};
        }

        macro_rules! read_array {
            ($file:expr, $type:ty) => {{
                // create a bytes buffer big enough for $len elements of type $type
                let len = read!($file, u64) as usize;
                let bytes = len * core::mem::size_of::<$type>();
                let mut buffer: Vec<u8> = vec![0; bytes];
                // read the file in the buffer
                $file.read_exact(&mut buffer)?;
                // convert the buffer Vec<u8> into a Vec<$type>
                let ptr = buffer.as_mut_ptr();
                core::mem::forget(buffer);
                unsafe { Vec::from_raw_parts(ptr as *mut $type, len, len) }
            }};
        }
        // actually lod the data :)
        let size = read!(file, u64);
        let multiplier = read!(file, u64);
        let global_seed = read!(file, u64);
        let edge_offset_and_seed = read_array!(file, u64);
        let array = read_array!(file, u64);

        Ok(Self {
            size,
            multiplier,
            global_seed,
            edge_offset_and_seed,
            array,
        })
    }
}
