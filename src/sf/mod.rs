/*
 *
 * SPDX-FileCopyrightText: 2023 Tommaso Fontana
 * SPDX-FileCopyrightText: 2023 Inria
 * SPDX-FileCopyrightText: 2023 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR LGPL-2.1-or-later
 */

//! # Static functions
//!
//! Static functions are data structures designed to store arbitrary mappings f : X → Σ from a finite
//! set of keys X ⊆ U to an alphabet Σ, where U is a universe of possible keys. Given x ∈ U, a static
//! function returns a value that is f(x) when x ∈ X; on U \ X, the result is irrelevant. In general,
//! one looks for representations returning their output in constant time.
//! Functions can be easily implemented using hash tables, but since they are allowed to return any
//! value if the queried key is not in X, in the static case we can break the information-theoretical lower
//! bound of storing the set X. In fact, constructions for static functions achieve just O(|X| log |Σ|)
//! bits of space, regardless of the nature of X and U. This makes static functions a powerful
//! techniques when handling, for instance, large sets of strings, and they are important building
//! blocks of space-efficient data structures such as (compressed) full-text indexes, (monotone)
//! MPHFs, Bloom filter-like data structures, and prefix-search data structures.

pub mod gov3;
