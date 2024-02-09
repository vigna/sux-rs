/*
 * SPDX-FileCopyrightText: 2023 Tommaso Fontana
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

use crate::{
    bits::{AtomicBitFieldVec, BitFieldVec},
    prelude::ConvertTo,
    traits::{AtomicBitFieldSlice, BitFieldSlice, BitFieldSliceCore, BitFieldSliceMut},
};
use anyhow::{ensure, Result};
use core::hash::{Hash, Hasher};
use core::marker::PhantomData;
use core::sync::atomic::{AtomicUsize, Ordering};
use epserde::prelude::*;
use mem_dbg::{MemDbg, MemSize};

mod bias;
mod tables;

/// This trait extends Hasher with a static method to build a new hasher.
pub trait HashStrategy {
    fn hash(value: &impl Hash) -> u64;
}

#[derive(Epserde, Debug, Clone, MemDbg, MemSize)]
pub struct DefaultStrategy;

impl HashStrategy for DefaultStrategy {
    #[deny(unconditional_recursion)]
    fn hash(value: &impl Hash) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        value.hash(&mut hasher);
        hasher.finish()
    }
}

#[derive(Epserde, Debug, Clone, MemDbg, MemSize)]
/// A Vector of HyperLogLog Counters, this is preferable to a standard vector
/// of hyperloglog counters as it stores the metadata just once and the registers
/// can be stored contiguosly without any padding.
///
/// If you need just a single hyperloglog counter use: <https://github.com/LucaCappelletti94/hyperloglog-rs>
pub struct HyperLogLogVec<
    V: Hash = usize,
    H: HashStrategy = DefaultStrategy,
    B = BitFieldVec<usize>,
> {
    /// A struct that stores a sequences of integers which we will use to store
    /// registers
    regs: B,
    /// The number of registers for each counter
    precision: usize,
    /// log2(precision)
    log2_precision: usize,
    /// A mask with the lowest log2(precision) bits set to 1, and everything else to 0
    precision_mask: u64,
    /// Multiplicative bias correction of the
    alpha: f32,
    /// The threshold used for when to enable the small values bias correction
    linear_counting_threshold: f32,
    /// A table of pre-computed corrections
    small_corrections: Vec<f32>,
    /// This is not really needed, but enforces that you can only insert
    /// data of the same type
    _marker: PhantomData<(V, H)>,
}

#[derive(Epserde, Debug, Clone, MemDbg, MemSize)]
/// A Vector of HyperLogLog Counters, this is preferable to a standard vector
/// of hyperloglog counters as it stores the metadata just once and the registers
/// can be stored contiguosly without any padding.
///
/// If you need just a single hyperloglog counter use: <https://github.com/LucaCappelletti94/hyperloglog-rs>
pub struct AtomicHyperLogLogVec<
    V: Hash = usize,
    H: HashStrategy = DefaultStrategy,
    B = AtomicBitFieldVec<usize>,
> {
    /// A struct that stores a sequences of integers which we will use to store
    /// registers
    regs: B,
    /// The number of registers for each counter
    precision: usize,
    /// log2(precision)
    log2_precision: usize,
    /// A mask with the lowest log2(precision) bits set to 1, and everything else to 0
    precision_mask: u64,
    /// Multiplicative bias correction of the
    alpha: f32,
    /// The threshold used for when to enable the small values bias correction
    linear_counting_threshold: f32,
    /// A table of pre-computed corrections
    small_corrections: Vec<f32>,
    /// This is not really needed, but enforces that you can only insert
    /// data of the same type
    _marker: PhantomData<(V, H)>,
}

impl<V: Hash, H: HashStrategy> ConvertTo<AtomicHyperLogLogVec<V, H>> for HyperLogLogVec<V, H> {
    fn convert_to(self) -> Result<AtomicHyperLogLogVec<V, H>> {
        Ok(AtomicHyperLogLogVec {
            regs: self.regs.convert_to()?,
            precision: self.precision,
            log2_precision: self.log2_precision,
            precision_mask: self.precision_mask,
            alpha: self.alpha,
            linear_counting_threshold: self.linear_counting_threshold,
            small_corrections: self.small_corrections,
            _marker: PhantomData,
        })
    }
}

impl<V: Hash, H: HashStrategy> ConvertTo<HyperLogLogVec<V, H>> for AtomicHyperLogLogVec<V, H> {
    fn convert_to(self) -> Result<HyperLogLogVec<V, H>> {
        Ok(HyperLogLogVec {
            regs: self.regs.convert_to()?,
            precision: self.precision,
            log2_precision: self.log2_precision,
            precision_mask: self.precision_mask,
            alpha: self.alpha,
            linear_counting_threshold: self.linear_counting_threshold,
            small_corrections: self.small_corrections,
            _marker: PhantomData,
        })
    }
}

impl<V: Hash> HyperLogLogVec<V> {
    /// Create a new HyperLogLogVec.
    ///
    /// # Arguments
    /// - `bits`: how many bits each register will take
    /// - `precision`: log2 of the number of registers per counter,
    ///     this is currently supported only up to 18.
    /// - `len`: the number of counters
    pub fn new(bits: usize, log2_precision: usize, len: usize) -> Result<Self> {
        ensure!(bits != 0, "Bits of hyperloglog regs cannot be 0");
        ensure!(bits <= 64, "Bits of hyperloglog regs cannot be over 64");
        Self::new_with(
            log2_precision,
            BitFieldVec::new(bits, (1 << log2_precision) * len),
        )
    }
}

impl<V: Hash> AtomicHyperLogLogVec<V> {
    /// Create a new HyperLogLogVec.
    ///
    /// # Arguments
    /// - `bits`: how many bits each register will take
    /// - `precision`: log2 of the number of registers per counter,
    ///     this is currently supported only up to 18.
    /// - `len`: the number of counters
    pub fn new(bits: usize, log2_precision: usize, len: usize) -> Result<Self> {
        ensure!(bits != 0, "Bits of hyperloglog regs cannot be 0");
        ensure!(bits <= 64, "Bits of hyperloglog regs cannot be over 64");
        Self::new_with(
            log2_precision,
            AtomicBitFieldVec::new(bits, (1 << log2_precision) * len),
        )
    }
}

impl<V: Hash, H: HashStrategy, B: BitFieldSliceCore<usize>> HyperLogLogVec<V, H, B> {
    /// Create a new HyperLogLogVec.
    ///
    /// # Arguments
    /// - `precision`: log2 of the number of registers per counter,
    ///     this is currently supported only up to 18.
    /// - `regs`: the regs to use
    pub fn new_with(log2_precision: usize, regs: B) -> Result<Self> {
        ensure!(
            log2_precision >= 4,
            "Precisions under 4 are not supported yet!"
        );
        ensure!(
            log2_precision <= 18,
            "Precisions over 18 are not supported yet!"
        );
        let precision: usize = 1 << log2_precision;
        let alpha = match log2_precision {
            4 => 0.673,
            5 => 0.697,
            6 => 0.709,
            _ => 0.7213 / (1.0 + 1.079 / precision as f32),
        };
        let linear_counting_threshold = match log2_precision {
            4 => 10.0,
            5 => 20.0,
            6 => 40.0,
            7 => 80.0,
            8 => 220.0,
            9 => 400.0,
            10 => 900.0,
            11 => 1800.0,
            12 => 3100.0,
            13 => 6500.0,
            14 => 11500.0,
            15 => 20000.0,
            16 => 50000.0,
            17 => 120000.0,
            18 => 350000.0,
            // the paper only has from 4 to 18.
            _ => unreachable!(),
        };
        Ok(Self {
            log2_precision,
            precision_mask: (precision as u64) - 1,
            precision,
            alpha,
            linear_counting_threshold,
            small_corrections: precompute_linear_counting(precision),
            regs,
            _marker: PhantomData,
        })
    }

    #[inline]
    /// Get the raw fixed point sum of 1/2**reg and the number of regs equal to zero
    /// and apply all the corrections, returning the estimate
    fn adjust_estimate(&self, raw_estimate: u64, zero_regs: usize) -> f32 {
        // apply small range corrections
        if zero_regs > 0 {
            let low_range_correction = self.small_corrections[zero_regs - 1];
            if low_range_correction <= self.linear_counting_threshold {
                return low_range_correction;
            }
        }

        // convert from fixed point to f32
        let exp = 1 + raw_estimate.leading_zeros() as u64;
        // set the fraction from bits 22 to 0 (23 bits)
        let mut mantissa = raw_estimate << exp;
        mantissa >>= 64 - 23;
        // set the exponent from bit 30 to 23 (8 bits)
        let exp = (127 + 64 - 32 - exp) << 23;
        // transmute this safely
        let raw_estimate = mantissa | exp;
        let raw_estimate = f32::from_ne_bytes((raw_estimate as u32).to_ne_bytes());

        // apply the correction
        let estimate = self.alpha * ((self.precision as f32).powi(2) / raw_estimate);

        // apply mid range corrections
        if estimate <= (5 * self.precision) as f32 {
            let biases = bias::DATA[self.log2_precision - 4];
            let estimates = tables::DATA[self.log2_precision - 4];

            // handle first bias
            if estimate <= estimates[0] {
                return estimate - biases[0];
            }
            // handle last bias
            if estimates[estimates.len() - 1] <= estimate {
                return estimate - biases[biases.len() - 1];
            }

            // Raw estimate is somewhere in between estimates.
            // Binary search for the calculated raw estimate.
            //
            // Here we unwrap because neither the values in `estimates`
            // nor `raw` are going to be NaN.
            let partition_index = estimates.partition_point(|est| *est <= estimate);

            // Return linear interpolation between raw's neighboring points.
            let ratio = (estimate - estimates[partition_index - 1])
                / (estimates[partition_index] - estimates[partition_index - 1]);

            // Calculate bias.
            return estimate
                - (biases[partition_index - 1]
                    + ratio * (biases[partition_index] - biases[partition_index - 1]));
        }

        estimate
    }
}

impl<V: Hash, H: HashStrategy, B: AtomicBitFieldSlice<usize>> AtomicHyperLogLogVec<V, H, B> {
    /// Create a new HyperLogLogVec.
    ///
    /// # Arguments
    /// - `bits`: how many bits each register will take
    /// - `precision`: log2 of the number of registers per counter,
    ///     this is currently supported only up to 18.
    /// - `regs`: the regs to use
    pub fn new_with(log2_precision: usize, regs: B) -> Result<Self> {
        ensure!(
            log2_precision >= 4,
            "Precisions under 4 are not supported yet!"
        );
        ensure!(
            log2_precision <= 18,
            "Precisions over 18 are not supported yet!"
        );
        let precision: usize = 1 << log2_precision;
        let alpha = match log2_precision {
            4 => 0.673,
            5 => 0.697,
            6 => 0.709,
            _ => 0.7213 / (1.0 + 1.079 / precision as f32),
        };
        let linear_counting_threshold = match log2_precision {
            4 => 10.0,
            5 => 20.0,
            6 => 40.0,
            7 => 80.0,
            8 => 220.0,
            9 => 400.0,
            10 => 900.0,
            11 => 1800.0,
            12 => 3100.0,
            13 => 6500.0,
            14 => 11500.0,
            15 => 20000.0,
            16 => 50000.0,
            17 => 120000.0,
            18 => 350000.0,
            // the paper only has from 4 to 18.
            _ => unreachable!(),
        };
        Ok(Self {
            log2_precision,
            precision_mask: (precision as u64) - 1,
            precision,
            alpha,
            linear_counting_threshold,
            small_corrections: precompute_linear_counting(precision),
            regs,
            _marker: PhantomData,
        })
    }

    #[inline]
    /// Get the raw fixed point sum of 1/2**reg and the number of regs equal to zero
    /// and apply all the corrections, returning the estimate
    fn adjust_estimate(&self, raw_estimate: u64, zero_regs: usize) -> f32 {
        // apply small range corrections
        if zero_regs > 0 {
            let low_range_correction = self.small_corrections[zero_regs - 1];
            if low_range_correction <= self.linear_counting_threshold {
                return low_range_correction;
            }
        }

        // convert from fixed point to f32
        let exp = 1 + raw_estimate.leading_zeros() as u64;
        // set the fraction from bits 22 to 0 (23 bits)
        let mut mantissa = raw_estimate << exp;
        mantissa >>= 64 - 23;
        // set the exponent from bit 30 to 23 (8 bits)
        let exp = (127 + 64 - 32 - exp) << 23;
        // transmute this safely
        let raw_estimate = mantissa | exp;
        let raw_estimate = f32::from_ne_bytes((raw_estimate as u32).to_ne_bytes());

        // apply the correction
        let estimate = self.alpha * ((self.precision as f32).powi(2) / raw_estimate);

        // apply mid range corrections
        if estimate <= (5 * self.precision) as f32 {
            let biases = bias::DATA[self.log2_precision - 4];
            let estimates = tables::DATA[self.log2_precision - 4];

            // handle first bias
            if estimate <= estimates[0] {
                return estimate - biases[0];
            }
            // handle last bias
            if estimates[estimates.len() - 1] <= estimate {
                return estimate - biases[biases.len() - 1];
            }

            // Raw estimate is somewhere in between estimates.
            // Binary search for the calculated raw estimate.
            //
            // Here we unwrap because neither the values in `estimates`
            // nor `raw` are going to be NaN.
            let partition_index = estimates.partition_point(|est| *est <= estimate);

            // Return linear interpolation between raw's neighboring points.
            let ratio = (estimate - estimates[partition_index - 1])
                / (estimates[partition_index] - estimates[partition_index - 1]);

            // Calculate bias.
            return estimate
                - (biases[partition_index - 1]
                    + ratio * (biases[partition_index] - biases[partition_index - 1]));
        }

        estimate
    }
}

impl<V: Hash, H: HashStrategy, B: BitFieldSlice<usize>> HyperLogLogVec<V, H, B> {
    /// Return an iterator over the registers of the counter with index `idx`
    pub fn iter_regs(&self, idx: usize) -> impl Iterator<Item = usize> + '_ {
        // get the start and end indices of the regs of this counter
        let start = idx * self.precision;
        let end = start + self.precision;
        // TODO!: use a real iterator, this will improve speed
        (start..end).map(|idx| self.regs.get(idx))
    }

    pub fn estimate_cardinality(&self, idx: usize) -> f32 {
        // fixed point float for speed and precision
        let mut raw_estimate: u64 = 0;
        let mut zero_regs = 0;
        for reg in self.iter_regs(idx) {
            // keep track of the regs that are at zero
            zero_regs += (reg == 0) as usize;
            // += 2**(-reg)
            raw_estimate += 1 << (32 - reg);
        }
        self.adjust_estimate(raw_estimate, zero_regs)
    }
}

impl<V: Hash, H: HashStrategy, B: AtomicBitFieldSlice<usize> + BitFieldSliceCore<AtomicUsize>>
    AtomicHyperLogLogVec<V, H, B>
{
    /// Return an iterator over the registers of the counter with index `idx`
    pub fn iter_regs(&self, idx: usize, order: Ordering) -> impl Iterator<Item = usize> + '_ {
        // get the start and end indices of the regs of this counter
        let start = idx * self.precision;
        let end = start + self.precision;
        // TODO!: use a real iterator, this will improve speed
        (start..end).map(move |idx| self.regs.get_atomic(idx, order))
    }

    pub fn estimate_cardinality(&self, idx: usize, order: Ordering) -> f32 {
        // fixed point float for speed and precision
        let mut raw_estimate: u64 = 0;
        let mut zero_regs = 0;
        for reg in self.iter_regs(idx, order) {
            // keep track of the regs that are at zero
            zero_regs += (reg == 0) as usize;
            // += 2**(-reg)
            raw_estimate += 1 << (32 - reg);
        }
        self.adjust_estimate(raw_estimate, zero_regs)
    }
}

impl<V: Hash, H: HashStrategy, B: BitFieldSlice<usize> + BitFieldSliceMut<usize>>
    HyperLogLogVec<V, H, B>
{
    /// Insert a value in the counter of index `idx`
    pub fn insert(&mut self, idx: usize, value: &V) {
        // compute the hash of the value
        let hash = H::hash(value);
        let base_idx = idx * self.precision;
        // get the reg_idx
        let reg_idx = base_idx + (hash & self.precision_mask) as usize;
        // count how many zeros we have from the MSB
        // because LZCOUNT is more supported and this way we don't need to clean
        // the lower bits
        let number_of_zeros =
            (1 + hash.leading_zeros() as usize).min((1 << self.regs.bit_width()) - 1);
        // get the old value
        let old_value = self.regs.get(reg_idx);
        // store the current value only if bigger
        if number_of_zeros > old_value {
            self.regs.set(reg_idx, number_of_zeros);
        }
    }

    /// Write the registers from an iterator to the counter with index `idx`
    pub fn from_iter(&mut self, idx: usize, mut regs_iter: impl Iterator<Item = usize>) {
        let start = idx * self.precision;
        let end = start + self.precision;
        for reg_idx in start..end {
            self.regs.set(reg_idx, regs_iter.next().unwrap());
        }
        debug_assert!(regs_iter.next().is_none());
    }

    /// Set all registers to zero
    pub fn reset(&mut self) {
        self.regs.reset()
    }
}

impl<V: Hash, H: HashStrategy, B: AtomicBitFieldSlice<usize>> AtomicHyperLogLogVec<V, H, B> {
    /// Insert a value in the counter of index `idx`
    pub fn insert(&self, idx: usize, value: &V, order: Ordering) {
        // compute the hash of the value
        let hash = H::hash(value);
        let base_idx = idx * self.precision;
        // get the reg_idx
        let reg_idx = base_idx + (hash & self.precision_mask) as usize;
        // count how many zeros we have from the MSB
        // because LZCOUNT is more supported and this way we don't need to clean
        // the lower bits
        let number_of_zeros =
            (1 + hash.leading_zeros() as usize).min((1 << self.regs.bit_width()) - 1);
        // get the old value
        let old_value = self.regs.get_atomic(reg_idx, order);
        // store the current value only if bigger
        if number_of_zeros > old_value {
            self.regs.set_atomic(reg_idx, number_of_zeros, order);
        }
    }

    /// Write the registers from an iterator to the counter with index `idx`
    pub fn from_iter(
        &self,
        idx: usize,
        mut regs_iter: impl Iterator<Item = usize>,
        order: Ordering,
    ) {
        let start = idx * self.precision;
        let end = start + self.precision;
        for reg_idx in start..end {
            self.regs
                .set_atomic(reg_idx, regs_iter.next().unwrap(), order);
        }
        debug_assert!(regs_iter.next().is_none());
    }

    /// Set all registers to zero
    pub fn reset(&mut self, order: Ordering) {
        self.regs.reset_atomic(order)
    }
}

/// Precompute a table of coefficients for faster linear counting
pub fn precompute_linear_counting(precision: usize) -> Vec<f32> {
    let mut small_corrections = vec![0_f32; precision];
    let mut i = 0;
    // We can skip the last value in the small range correction array, because it is always 0.
    while i < precision - 1 {
        small_corrections[i] = (precision as f64 * (precision as f64 / (i + 1) as f64).ln()) as f32;
        i += 1;
    }
    small_corrections
}
