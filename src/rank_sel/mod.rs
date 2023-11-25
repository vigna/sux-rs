/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Support for rank and select operations.

mod select_fixed1;
pub use select_fixed1::*;

mod select_zero_fixed1;
pub use select_zero_fixed1::*;

mod select_fixed2;
pub use select_fixed2::*;

mod select_zero_fixed2;
pub use select_zero_fixed2::*;
