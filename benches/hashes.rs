#![feature(test)]
#![allow(deprecated)]

extern crate test;

use core::hash::Hasher;
use test::{black_box, Bencher};

macro_rules! bench_hash {
    ($ty:ty, $test_name:ident) => {
        #[bench]
        fn $test_name(b: &mut Bencher) {
            let data = std::fs::read_to_string("tests/data/wordlist.10000").unwrap();
            let words = data.split_whitespace().collect::<Vec<_>>();
            b.bytes = data.len() as u64;
            b.iter(|| {
                for string in words.iter() {
                    let mut hasher = <$ty>::default();
                    hasher.write(black_box(string.as_bytes()));
                    let _ = black_box(hasher.finish());
                }
            })
        }
    };
}

bench_hash!(
    std::collections::hash_map::DefaultHasher,
    bench_default_hasher
);

bench_hash!(core::hash::SipHasher, bench_std_siphasher);

//#[cfg(target_feature = "sse")]
//bench_hash!(sux::hash::sip::SipHash64SSE<2, 4>, bench_siphash_sse_24);
//#[cfg(target_feature = "sse")]
//bench_hash!(sux::hash::sip::SipHash64SSE<1, 3>, bench_siphash_sse_14);
//#[cfg(target_feature = "sse")]
//bench_hash!(sux::hash::sip::SipHash64SSE<4, 8>, bench_siphash_sse_48);
