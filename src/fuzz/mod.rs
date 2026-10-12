/*
 * SPDX-FileCopyrightText: 2023 Tommaso Fontana
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

//! Fuzzing harnesses, enabled by the `fuzz` feature.
//!
//! The harnesses are used by the fuzz targets in the `fuzz` directory of the
//! repository, and they can also be called directly in tests.

pub mod select;
