/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Indexed dictionaries.

pub mod elias_fano;
pub mod rear_coded_list;

pub mod prelude {
    pub use super::elias_fano::*;
    pub use super::rear_coded_list::*;
}
