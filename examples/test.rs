use rand::rngs::SmallRng;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::rand_core::RngCore;
use rand_chacha::ChaCha8Rng;
use std::hint::black_box;
use std::time::Instant;

fn main() {
    const N: usize = 1000000000;

    let start = Instant::now();
    let mut r = StdRng::seed_from_u64(0);
    for _ in 0..N {
        black_box(r.gen::<u64>());
    }

    println!("ChaCha12 {}", start.elapsed().as_nanos() as f64 / N as f64);

    let start = Instant::now();
    let mut r = ChaCha8Rng::seed_from_u64(0);
    for _ in 0..N {
        black_box(r.next_u64());
    }

    println!("ChaCha8 {}", start.elapsed().as_nanos() as f64 / N as f64);

    let start = Instant::now();
    let mut r = SmallRng::seed_from_u64(0);
    for _ in 0..N {
        black_box(r.gen::<u64>());
    }

    println!(
        "xoshiro256++ {}",
        start.elapsed().as_nanos() as f64 / N as f64
    );
}
