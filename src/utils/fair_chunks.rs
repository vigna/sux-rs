/*
 * SPDX-FileCopyrightText: 2025 Inria
 * SPDX-FileCopyrightText: 2025 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use crate::traits::{Succ, SuccUnchecked};

/// An iterator returning fair chunks.
///
/// Given positive integer weights on the first `n` integers, this iterator
/// returns chunks of integers (in the form of ranges) such that the sum of the
/// weights in each chunk is approximately the same, and approximately equal to
/// a provided target weight.
///
/// To build an iterator, you need to provide the cumulative function of weights
/// (i.e., the sequence 0, `w₀`, `w₀ + w₁`, `w₀ + w₁ + w₂`, …), stored in a
/// structure that supports [(unchecked) successor queries](crate::traits::Succ).
///
/// # Example
///
/// ```rust
/// use sux::prelude::*;
/// use sux::utils::FairChunks;
/// // The weights of our elements
/// let weights = [
///     15, 27, 20, 26,  4, 22, 10, 25, 7, 13,  0, 11, 5, 28, 23,
///     1, 12, 24,  3, 30,  8, 29, 17, 2, 14,  9, 16, 18, 21, 19,
/// ];
/// // Compute the cumulative weight function
/// let mut cwf = vec![0];
/// cwf.extend(weights.iter().scan(0, |acc, x| {
///    *acc += x;
///   Some(*acc)
/// }));
/// // Put the sequence in an Elias–Fano structure
/// let mut efb = EliasFanoBuilder::new(cwf.len(), *cwf.last().unwrap());
/// efb.extend(cwf);
/// let ef = efb.build_with_seq_and_dict();
/// // Create the iterator
/// let chunks = FairChunks::new(50, &ef);
/// // The weight of the chunks is balanced
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
///
/// // To save memory, we can build a smaller Elias–Fano structure
/// // only supporting unchecked successor queries
/// let mut cwf = vec![0];
/// cwf.extend(weights.iter().scan(0, |acc, x| {
///    *acc += x;
///   Some(*acc)
/// }));
/// let last_cwf = *cwf.last().unwrap();
/// let mut efb = EliasFanoBuilder::new(cwf.len(), last_cwf);
/// efb.extend(cwf);
/// let ef = efb.build_with_dict();
/// let chunks = FairChunks::new_with(50, &ef, weights.len(), last_cwf);
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
pub struct FairChunks<I: SuccUnchecked<Input = usize, Output = usize>> {
    /// Cumulative weight function. This is used to generate chunks with
    /// approximately the same target weight.
    cwf: I,
    /// How much overall weight each chunk will approximately have. When 0,
    /// the iterator is exhausted.
    target_weight: usize,
    /// The position of the first non-returned element.
    curr_pos: usize,
    /// The weight at [`start_pos`](Self::curr_pos).
    current_weight: usize,
    /// The number of weights.
    num_weights: usize,
    /// The last element of [cwf](Self::cwf).
    max_weight: usize,
}

impl<I: SuccUnchecked<Input = usize, Output = usize>> FairChunks<I> {
    /// Create a fair chunk iterator using a structure supporting unchecked
    /// successor queries.
    ///
    /// This constructor does not require the cumulative weight function to
    /// implement [`Succ`], but rather just [`SuccUnchecked`]. In the typical
    /// case of [`EliasFano`](crate::dict::EliasFano), it is sufficient to have
    /// selection on zeros.
    ///
    /// # Arguments
    ///
    /// * `target_weight` - The target weight of the chunks.
    ///
    /// * `cwf` - The cumulative weight function.
    ///
    /// * `num_weights` - The number of weights.
    ///
    /// * `max_weight` - The last element of the cumulative weight function.
    pub fn new_with(target_weight: usize, cwf: I, num_weights: usize, max_weight: usize) -> Self {
        Self {
            target_weight,
            cwf,
            curr_pos: 0,
            current_weight: 0,
            num_weights,
            max_weight,
        }
    }
}

impl<I: Succ<Input = usize, Output = usize>> FairChunks<I> {
    /// Create a fair chunk iterator using a structure supporting successor
    /// queries.
    ///
    /// This constructor requires that the cumulative weight function implements
    /// implement [`Succ`]. In the typical case of
    /// [`EliasFano`](crate::dict::EliasFano), it is necessary to have
    /// selections on zeroes and ones. The constructor
    /// [`new_with`](Self::new_with) makes it possible to use a cumulative
    /// weight function that only implements [`SuccUnchecked`].
    ///
    /// # Arguments
    ///
    /// * `target_weight` - The target weight of the chunks.
    ///
    /// * `cwf` - The cumulative weight function.
    pub fn new(target_weight: usize, cwf: I) -> Self {
        let len = cwf.len();
        let max_weight = if len == 0 { 0 } else { cwf.get(len - 1) };
        Self {
            target_weight,
            cwf,
            curr_pos: 0,
            current_weight: 0,
            num_weights: len - 1,
            max_weight,
        }
    }
}

impl<I: SuccUnchecked<Input = usize, Output = usize>> Iterator for FairChunks<I> {
    type Item = core::ops::Range<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.target_weight == 0 {
            return None;
        }

        let target = self.current_weight + self.target_weight;
        Some(if target > self.max_weight {
            self.target_weight = 0;
            self.curr_pos..self.num_weights
        } else {
            let (next_pos, next_weight) = unsafe { self.cwf.succ_unchecked::<false>(&target) };
            self.current_weight = next_weight;
            let res = self.curr_pos..next_pos;
            self.curr_pos = next_pos;
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
        // The weights of our elements
        let weights = [
            15, 27, 20, 26, 4, 22, 10, 25, 7, 13, 0, 11, 5, 28, 23, 1, 12, 24, 3, 30, 8, 29, 17, 2,
            14, 9, 16, 18, 21, 19,
        ];
        // Compute the cumulative weight function
        let mut cwf = vec![0];
        cwf.extend(weights.iter().scan(0, |acc, x| {
            *acc += x;
            Some(*acc)
        }));
        // Put the sequence in an Elias–Fano structure
        let mut efb = EliasFanoBuilder::new(cwf.len(), *cwf.last().unwrap());
        efb.extend(cwf);
        let ef = efb.build_with_seq_and_dict();
        // Create the iterator
        let chunks = FairChunks::new(threshold, &ef);
        let chunks_weights = chunks
            .map(|x| x.map(|i| weights[i]).sum::<usize>())
            .collect::<Vec<_>>();
        assert!(
            chunks_weights
                .iter()
                .take(chunks_weights.len() - 2)
                .all(|x| *x >= threshold),
            "All chunks, except the last one, must have weight over the threshold."
        );
        // check that without the last element in the range the weight is less
        // than the threshold
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
