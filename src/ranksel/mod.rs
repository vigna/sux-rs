/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Rank and select operations.

pub mod elias_fano;
pub mod sparse_index;
pub mod sparse_zero_index;

pub mod prelude {
    pub use super::elias_fano::*;
    pub use super::sparse_index::*;
    pub use super::sparse_zero_index::*;
}
