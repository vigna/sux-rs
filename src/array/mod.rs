/*
 * SPDX-FileCopyrightText: 2024 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Array data structures.
//!
//! This module provides array-like data structures with specialized
//! characteristics for succinct data structure applications.

pub mod partial_array;
pub use partial_array::{PartialArray, PartialArrayBuilder, PartialArrayIter};
