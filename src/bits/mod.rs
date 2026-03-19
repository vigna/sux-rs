/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Structures for [bit vectors](`mod@bit_vec`) and [vectors of bit fields of
//! fixed width](`mod@bit_field_vec`).
//!
//! These are the foundational storage types on which [rank/select
//! structures](crate::rank_sel) and [indexed dictionaries](crate::dict) are
//! built. The [`bit_vec!`](crate::bit_vec) and
//! [`bit_field_vec!`](crate::bit_field_vec) macros provide convenient
//! construction.
//!
//! Bit vectors and bit-field vectors are parameterized by a
//! [`Backend`](crate::traits::Backend), which abstracts over different storage
//! types (`Vec<W>`, `Box<[W]>`, memory-mapped slices, etc.). They also
//! implement the [`Backend`] trait themselves by delegation.
//!
//! In particular, any [rank/select structure](crate::rank_sel) requires a
//! [`Backend`] and is a [`Backend`](crate::traits::Backend) itself by
//! delegation, so that every structure can be used as the backend of a further
//! structure.

pub mod bit_field_vec;
pub use bit_field_vec::*;

pub mod bit_vec;
pub use bit_vec::*;

pub use crate::bit_field_vec;
pub use crate::bit_vec;
