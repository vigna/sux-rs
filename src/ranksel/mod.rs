/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Rank and select operations.

pub mod quantum_index;
pub mod quantum_zero_index;

pub mod prelude {
    pub use super::quantum_index::*;
    pub use super::quantum_zero_index::*;
}
