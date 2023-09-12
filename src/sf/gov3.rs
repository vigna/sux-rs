/*
 *
 * SPDX-FileCopyrightText: 2023 Tommaso Fontana
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Ported from <https://github.com/vigna/Sux4J/blob/master/c/sf3.c>

use crate::hash::spooky::{spooky_short, spooky_short_rehash};
use anyhow::Result;
use std::fs::File;
use std::io::BufReader;
use std::io::Read;
use std::path::Path;

#[derive(Debug)]
/// An immutable function stored quasi-succinctly using the Linear3SystemSolver
/// Genuzio-Ottaviano-Vigna method to solve **F**2-linear systems.
///
/// Static function structure that reads Java-generated, dumped structures.
///
/// To generate the structure, you can use the Java version as:
/// ```bash
/// java -Xmx500G $JVMOPTS it.unimi.dsi.sux4j.mph.GOV3Function 7B.64-gov 7B.out -b
/// ```
///
/// To obtain a file that can be read by this structure, load
/// the Java instance of the static function and write it to a file
/// using the `dump` method.
///
/// You can do it through:
/// ```shell
/// echo '((it.unimi.dsi.sux4j.mph.GOV3Function)it.unimi.dsi.fastutil.io.BinIO.loadObject("test.mph")).dump("test.cmph");' | jshell
/// ```
///
/// # Reference:
/// [Marco Genuzio, Giuseppe Ottaviano, and Sebastiano Vigna, Fast Scalable Construction of (Minimal Perfect Hash) Functions](https://arxiv.org/pdf/1603.04330.pdf)
/// [Java version with `dump` method](https://github.com/vigna/Sux4J/blob/master/src/it/unimi/dsi/sux4j/mph/GOV3Function.java)

pub struct GOV3 {
    pub size: u64,
    pub width: u64,
    pub multiplier: u64,
    pub global_seed: u64,
    pub offset_and_seed: Vec<u64>,
    pub array: Vec<u64>,
}

impl GOV3 {
    /// Given a path to a file `.csf3` generated from the java version,
    /// load a GOV3 structure from a file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::load_reader(BufReader::new(File::open(path.as_ref())?))
    }

    /// Given a generic `Read` implementor, load a GOV3 structure from a file.
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
        let width = read!(file, u64);
        let multiplier = read!(file, u64);
        let global_seed = read!(file, u64);
        let offset_and_seed = read_array!(file, u64);
        let array = read_array!(file, u64);

        Ok(Self {
            size,
            width,
            multiplier,
            global_seed,
            offset_and_seed,
            array,
        })
    }
}

impl GOV3 {
    pub fn size(&self) -> u64 {
        self.size
    }

    pub fn get_byte_array(&self, key: &[u8]) -> u64 {
        let signature = spooky_short(key, self.global_seed);
        let bucket = ((((signature[0] as u128) >> 1) * (self.multiplier as u128)) >> 64) as u64;
        let offset_seed = self.offset_and_seed[bucket as usize];
        let bucket_offset = offset_seed & OFFSET_MASK;
        let num_variables =
            (self.offset_and_seed[bucket as usize + 1] & OFFSET_MASK) - bucket_offset;
        let e = signature_to_equation(&signature, offset_seed & (!OFFSET_MASK), num_variables);
        get_value(&self.array, e[0] + bucket_offset, self.width)
            ^ get_value(&self.array, e[1] + bucket_offset, self.width)
            ^ get_value(&self.array, e[2] + bucket_offset, self.width)
    }
}

const OFFSET_MASK: u64 = u64::MAX >> 8;

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
fn get_value(array: &[u64], mut pos: u64, width: u64) -> u64 {
    pos *= width;
    let l = 64 - width;
    let start_word = pos / 64;
    let start_bit = pos % 64;
    if start_bit <= l {
        (array[start_word as usize] << (l - start_bit)) >> l
    } else {
        (array[start_word as usize] >> start_bit)
            | ((array[start_word as usize + 1] << (64 + l - start_bit)) >> l)
    }
}
