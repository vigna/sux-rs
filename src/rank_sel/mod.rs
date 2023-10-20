/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Support for rank and select operations.

mod quantum_index;
pub use quantum_index::*;

mod quantum_zero_index;
pub use quantum_zero_index::*;

//mod simple_select_half;
//pub use simple_select_half::*;
