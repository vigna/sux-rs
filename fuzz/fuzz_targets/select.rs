#![no_main]

use libfuzzer_sys::fuzz_target;
use sux::fuzz::select::{harness, Data};

fuzz_target!(|data: Data| harness(data));
