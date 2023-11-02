/*
 *
 * SPDX-FileCopyrightText: 2023 Tommaso Fontana
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

/*!

Static functions.

This module contains implementations of static functions, that is, immutable data structures
that store key/value pairs and allow to retrieve the value associated to a key.

Differently from a dictionary, static functions may return any result on a key that is not
part of the original set of keys. This property makes it possible to design static functions
using space very close to the space theoretical lower bound, which, for a function with `n` keys, and
a `b`-bit output, is `n * b + o(n)`.

The typical use case is that of a very large set of keys: the static function is built and
serialized, and used later.

*/
mod vfunc;
pub use vfunc::VFunc;
pub use vfunc::VFuncBuilder;
