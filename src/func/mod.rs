/*
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! Static functions.
//! 
//! Static functions map keys to values, but they do not store the keys:
//! querying a static function with a key outside of the original set will lead
//! to an arbitrary result.
//! 
//! In exchange, static functions have a very low space overhead, and make it
//! possible to store the association between keys and values just in the space
//! required by the values (with a small overhead).
//! 
//! See [`VFunc`] for more details.


pub mod vfunc;
pub use vfunc::*;

pub mod vbuilder;
pub use vbuilder::*;
