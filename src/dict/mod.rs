/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Indexed dictionaries.

pub mod elias_fano;
pub use elias_fano::{EliasFano, EliasFanoBuilder, EliasFanoConcurrentBuilder};

pub mod rear_coded_list;
pub use rear_coded_list::{RearCodedList, RearCodedListBuilder};

pub mod slice_seq;
pub use slice_seq::SliceSeq;
