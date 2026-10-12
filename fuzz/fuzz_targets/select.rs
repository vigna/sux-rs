/*
 * SPDX-FileCopyrightText: 2023 Tommaso Fontana
 *
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

#![no_main]

use libfuzzer_sys::fuzz_target;
use sux::fuzz::select::{harness, Data};

fuzz_target!(|data: Data| harness(data));
