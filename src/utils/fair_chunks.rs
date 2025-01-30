/*
 * SPDX-FileCopyrightText: 2025 Inria
 * SPDX-FileCopyrightText: 2025 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use crate::traits::Succ;

/// An iterator of elements with approximately the same "weight" over a generic
/// cumulative weight function that implements [`Succ`].
///
/// # Example
///
/// ```rust
/// use sux::prelude::*;
/// use sux::utils::FairChunks;
/// // the weights of our elements
/// let weights = [
///     15, 27, 20, 26,  4, 22, 10, 25, 7, 13,  0, 11, 5, 28, 23,
///     1, 12, 24,  3, 30,  8, 29, 17, 2, 14,  9, 16, 18, 21, 19,
/// ];
/// // prefix sum
/// let mut cwf = weights
/// .iter()
/// .scan(0, |acc, x| {
///     let res = *acc;
///     *acc += x;
///     Some(res)
/// })
/// .collect::<Vec<_>>();
/// cwf.push(cwf[29] + weights[29]); // last element! the CWF starts at zero
/// // put the sequence in an elias-fano
/// let mut efb = EliasFanoBuilder::new(cwf.len(), *cwf.last().unwrap());
/// efb.extend(cwf);
/// let ef = efb.build_with_seq_and_dict();
/// // create the iterator
/// let chunks = FairChunks::new(50, &ef);
/// // the weight of each chunks is over threshold, except for the last one.
/// assert_eq!(
///     chunks.collect::<Vec<_>>(),
///     vec![
///         0..3,   // weight 62
///         3..6,   // weight 52
///         6..10,  // weight 55
///         10..15, // weight 67
///         15..20, // weight 70
///         20..23, // weight 54
///         23..28, // weight 59
///         28..30, // weight 40
///     ],
/// );
/// ```
#[derive(Debug, Clone, Copy)]
pub struct FairChunks<I: Succ<Input = usize, Output = usize>> {
    /// Cumulative weight function. This is used to generate chunks with
    /// approximately the same weight.
    cwf: I,
    /// How much "weight" each chunk will approximately have. This is internally
    /// used to stop the iterator when set to 0 which is not a valid state.
    step: usize,
    /// The position of the first non-returned element
    start_pos: usize,
    /// The weight at `start_pos`
    current_weight: usize,
    /// The last element of CWF.
    max_weight: usize,
}

impl<I: Succ<Input = usize, Output = usize>> FairChunks<I> {
    /// Creates a new iterator that generates chunks with approximately the same
    /// weight.
    pub fn new(step: usize, cwf: I) -> Self {
        let max_weight = cwf.get(cwf.len() - 1);
        Self {
            step,
            cwf,
            start_pos: 0,
            current_weight: 0,
            max_weight,
        }
    }
}

impl<I: Succ<Input = usize, Output = usize>> Iterator for FairChunks<I> {
    type Item = core::ops::Range<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.step == 0 {
            return None;
        }

        let target = self.current_weight + self.step;
        Some(if target > self.max_weight {
            self.step = 0;
            self.start_pos..self.cwf.len() - 1
        } else {
            let (next_pos, next_weight) = self.cwf.succ(&target).unwrap();
            self.current_weight = next_weight;
            let res = self.start_pos..next_pos;
            self.start_pos = next_pos;
            res
        })
    }
}

#[cfg(test)]
mod test {
    use super::FairChunks;
    use crate::prelude::*;
    #[test]
    fn test_fair_chunks() {
        let threshold = 50;
        // the weights of our elements
        let weights = [
            15, 27, 20, 26, 4, 22, 10, 25, 7, 13, 0, 11, 5, 28, 23, 1, 12, 24, 3, 30, 8, 29, 17, 2,
            14, 9, 16, 18, 21, 19,
        ];
        // prefix sum
        let mut cwf = weights
            .iter()
            .scan(0, |acc, x| {
                let res = *acc;
                *acc += x;
                Some(res)
            })
            .collect::<Vec<_>>();
        cwf.push(cwf[29] + weights[29]);
        // put the sequence in an elias-fano
        let mut efb = EliasFanoBuilder::new(cwf.len(), *cwf.last().unwrap());
        efb.extend(cwf);
        let ef = efb.build_with_seq_and_dict();
        // create the iterator
        let chunks = FairChunks::new(threshold, &ef);
        let chunks_weights = chunks
            .map(|x| x.map(|i| weights[i]).sum::<usize>())
            .collect::<Vec<_>>();
        println!("{:?}", chunks_weights);
        assert!(
            chunks_weights
                .iter()
                .take(chunks_weights.len() - 2)
                .all(|x| *x >= threshold),
            "All chunks, except the last one, must have weight over the threshold."
        );
        // check that without the last element in the range, the weight is less than threshold.
        assert!(
            chunks
                .map(|x| (x.start..x.end - 1).map(|i| weights[i]).sum::<usize>())
                .all(|x| x < threshold),
            "All chunks, without their last element should be under the threshold."
        );

        let chunks = FairChunks::new(threshold, &ef);
        assert_eq!(
            chunks.collect::<Vec<_>>(),
            vec![0..3, 3..6, 6..10, 10..15, 15..20, 20..23, 23..28, 28..30,],
        );
    }
}
