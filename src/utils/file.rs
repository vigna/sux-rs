/*
 *
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

/*!

Utility wrappers for files.

*/

use std::{io::*, path::Path};
use zstd::stream::read::Decoder;

/// Adapter to iterate over the lines of a file.
#[derive(Clone)]
pub struct FilenameIntoIterator<P: AsRef<Path>>(pub P);

impl<P: AsRef<Path>> IntoIterator for FilenameIntoIterator<P> {
    type Item = String;
    type IntoIter = std::iter::Map<
        std::io::Lines<BufReader<std::fs::File>>,
        fn(std::io::Result<String>) -> String,
    >;

    fn into_iter(self) -> Self::IntoIter {
        BufReader::new(std::fs::File::open(self.0).unwrap())
            .lines()
            .map(|line| line.unwrap())
    }
}

/// Adapter to iterate over the lines of a file compressed with Zstandard.
#[derive(Clone)]
pub struct FilenameZstdIntoIterator<P: AsRef<Path>>(pub P);

impl<P: AsRef<Path>> IntoIterator for FilenameZstdIntoIterator<P> {
    type Item = String;
    type IntoIter = std::iter::Map<
        std::io::Lines<BufReader<Decoder<'static, BufReader<std::fs::File>>>>,
        fn(std::io::Result<String>) -> String,
    >;

    fn into_iter(self) -> Self::IntoIter {
        BufReader::new(Decoder::new(std::fs::File::open(self.0).unwrap()).unwrap())
            .lines()
            .map(|line| line.unwrap())
    }
}
