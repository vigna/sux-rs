/*
 * SPDX-FileCopyrightText: 2026 Sebastiano Vigna
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

//! Without the `rayon` feature a [`CompVFunc`] cannot be *built* (construction
//! goes through the rayon-only `VBuilder`), but the type and its query API must
//! still be available so a `CompVFunc` serialized elsewhere can be loaded and
//! queried. This target only needs to compile; before the fix the whole
//! `CompVFunc` type and export were gated behind `rayon`.

#![cfg(not(feature = "rayon"))]

use sux::func::CompVFunc;

fn query(f: &CompVFunc<u64>, key: u64) -> usize {
    f.get(key)
}

#[test]
fn comp_vfunc_type_and_query_available_without_rayon() {
    // The type is nameable and the query API type-checks without rayon.
    let _ = None::<CompVFunc<u64>>;
    let _query: fn(&CompVFunc<u64>, u64) -> usize = query;
}

#[cfg(feature = "epserde")]
fn assert_deserializable<T: epserde::deser::Deserialize>() {}

#[cfg(feature = "epserde")]
#[test]
fn comp_vfunc_deserializable_without_rayon() {
    // The whole point of ungating the type: a CompVFunc built (with rayon)
    // elsewhere and serialized must be loadable without rayon.
    assert_deserializable::<CompVFunc<u64>>();
}
